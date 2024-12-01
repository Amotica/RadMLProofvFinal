from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from dataset_helpers import *
from models import *
import os
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from keras import backend as K


def parameters_to_file(parameters, output_dir):
    with open(output_dir + "/training_parameters.txt", "w") as para_file:
        para_file.write(str(parameters))


def compute_confidence(num_classes, confidence_delta):
    """
    :param num_classes: the number of classes in the dataset
    :param confidence_delta: the small change to be added to the minimum winning probability
    Example, for three classes, the winning probability is 1/3 = 0.333 + small change (delta)
    :return: the cofidence value, which is minimum winning probability plus confidence_delta
    """
    # Prediction confidence depends on the num of classes
    max_win_prob = 1.0 / num_classes
    confidence_threshold = max_win_prob + confidence_delta

    return confidence_threshold


def train_baseline(parameters):
    train_dataset = parameters["train_dataset_baseline"]
    test_dataset = parameters["test_dataset"]
    batch_size=parameters["batch_size"]
    epochs = parameters["epochs"]
    modelsFolder = parameters["modelsFolder"]
    n_ratio = parameters["ratio_noisy"]
    # ===========================
    # train baseline with noise
    # ==========================

    x_train = train_dataset[:, :-1]  # all columns but the last three as they are labels
    y_train = train_dataset[:, -1]  # the labels (classes)
    x_train, y_train = shuffle(x_train, y_train)  # Shuffle for training

    x_test, y_test = test_dataset  # all columns but the last three as they are labels

    features, num_classes = get_features_and_classes(x_test, y_test)
    model = SNetModelDouble(features, num_classes)
    # =================================================================
    # create the ensemble of classifiers
    # =================================================================
    ensemble = [
        SNetModelDouble(features, num_classes),
        FNetModelDouble(features, num_classes),
        BasicFNetModel(features, num_classes)
    ]

    if not os.path.exists(modelsFolder):
        os.makedirs(modelsFolder, mode=0o777)

    for i, model in enumerate(ensemble):
        # check if the ensemble model files exists and load them up
        check_pt_file = modelsFolder + "/" + model.name + "_baseline_" + str(n_ratio) + ".h5"
        if os.path.isfile(check_pt_file):
            print("Loading the model weight file...")
            model.load_weights(check_pt_file)

        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.1,
            patience=10,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        )

        model_checkpoint = ModelCheckpoint(
            check_pt_file,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        # Compute the steps per epoch and validation steps
        steps_per_epoch = int(len(y_train) / batch_size)
        test_steps = int(len(y_test) / batch_size)

        history = model.fit(
            x_train, y_train,
            callbacks=[model_checkpoint, reduce_lr],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            validation_steps=test_steps
        )


def bootstrap_ensemble_learning(
        ensemble,
        X_labeled,
        y_labeled,
        subset_ratio,
        ratio_noisy,
        X_test,
        y_test,
        num_classes,
        modelsFolder,
        bootstrap_iter
):
    best_accuracy = 0.0
    best_model = None

    # Semi-Supervised Learning Loop: Ensemble co-training with confidence loop
    for iteration in range(bootstrap_iter):
        # get the random subset which is
        X_labeled_subset, y_labeled_subset = \
            random_data_subset(X_labeled, y_labeled, subset_ratio)
        print("    >>>Ensemble co-training iteration: #", iteration, " ...")
        print("    >>>================================================")
        print("Training on subset: ", len(y_labeled_subset))

        # Compute the steps per epoch and validation steps
        steps_per_epoch = int(len(y_labeled_subset) / batch_size)
        test_steps = int(len(y_test) / batch_size)

        if not os.path.exists(modelsFolder):
            os.makedirs(modelsFolder, mode=0o777)

        # Train each neural network on a random subset of the labeled data
        for i, model in enumerate(ensemble):
            print("      >>>Working on Ensemble Model: ", model.name, " ...")
            # reset the learning rate before training
            K.set_value(model.optimizer.learning_rate, 0.001)
            reduced_lr = ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.1,
                patience=10,
                verbose=1,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=0,
            )

            # check if the ensemble model files exists and load them up
            check_pt_file = modelsFolder + "/" + model.name + "_" + str(ratio_noisy) + ".h5"
            if os.path.isfile(check_pt_file):
                print("Loading the model weight file...")
                model.load_weights(check_pt_file)

            model_checkpoint = ModelCheckpoint(
                check_pt_file,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max'
            )

            history = model.fit(
                X_labeled_subset, y_labeled_subset,
                callbacks=[model_checkpoint, reduced_lr],
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                validation_steps=test_steps
            )

        # Prediction confidence depends on the num of classes
        confidence = compute_confidence(num_classes, confidence_delta)

        # Predictions from each neural network on the unlabeled data
        predictions = np.array([model.predict(X_test) for model in ensemble])

        # Confidence scores for each instance
        mean_predictions = np.mean(predictions, axis=0)
        mean_pred = np.argmax(mean_predictions, axis=1)  # index

        # Calculate the accuracy
        accuracy = accuracy_score(y_test, mean_pred)
        if accuracy > best_accuracy:
            # store the model to predict the unlabelled data after the bootstrap iteration
            best_model = ensemble
            best_accuracy = accuracy

    return best_model


def train_semi_supervised_ensemble(parameters):
    train_dataset = parameters["train_dataset_semi_sup"]
    test_dataset = parameters["test_dataset"]
    batch_size = parameters["batch_size"]
    modelsFolder = parameters["modelsFolder"]
    epochs = parameters["epochs"]
    confidence_delta = parameters["confidence_delta"]
    ratio_noisy = parameters["ratio_noisy"]
    subset_ratio = parameters["subset_ratio"]
    co_train_iter = parameters["co_train_iter"]
    bootstrap_iter = parameters["bootstrap_iter"]
    it_delta = parameters["it_delta"]
    # =================================================================
    # get the datasets numpy array and split them
    # =================================================================
    X_labeled, y_labeled, X_unlabeled, y_unlabeled = train_dataset
    X_test, y_test = test_dataset

    # =================================================================
    # create the ensemble of classifiers
    # =================================================================
    features, num_classes = get_features_and_classes(X_labeled, y_labeled)
    ensemble = [
        SNetModelDouble(features, num_classes),
        FNetModelDouble(features, num_classes),
        BasicFNetModel(features, num_classes)
    ]

    for iteration in range(co_train_iter):
        print("  >>>Starting Co-Learning Iteration: #", iteration, " ... ")
        print("  >>>================================================")

        # ==================================================
        # bootstrap ensemble learning to get the best model
        # ==================================================
        best_models = bootstrap_ensemble_learning(
            ensemble,
            X_labeled,
            y_labeled,
            subset_ratio,
            ratio_noisy,
            X_test,
            y_test,
            num_classes,
            modelsFolder,
            bootstrap_iter
        )

        # TODO: Check that y_unlabeled is less that it_delta percentage.
        #  if it is, the training stops
        stop_iter = int((y_labeled.size + y_unlabeled.size) * it_delta)
        if y_unlabeled.size == stop_iter:
            break

        # Prediction confidence depends on the num of classes
        confidence = compute_confidence(num_classes, confidence_delta)
        # Predictions from each neural network on the unlabeled data
        predictions = np.array([model.predict(X_unlabeled) for model in best_models])
        # Confidence scores for each instance
        mean_predictions = np.mean(predictions, axis=0)
        y_pred = np.argmax(mean_predictions, axis=1)  # index
        y_pred_prob = np.max(mean_predictions, axis=1)  # probability corresponding to the index
        confident_indices = np.where(y_pred_prob > confidence)[0]  # Identify probabilities with high confidence

        # Add confident instances to the labeled set
        X_labeled = np.vstack([X_labeled, X_unlabeled[confident_indices]])
        # The predicted labels for the unlabeled classes have been assigned,
        # as the model expresses high confidence in these predictions.
        y_labeled = np.concatenate([y_labeled, y_pred[confident_indices]])
        print("X_labeled ", X_labeled.shape)
        print("y_labeled ", y_labeled.shape)

        X_unlabeled = np.delete(X_unlabeled, confident_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, confident_indices, axis=0)
        print("X_unlabeled ", X_unlabeled.shape)
        print("y_unlabeled ", y_unlabeled.shape)

        # transfer learning from previously trained bootsrap sample
        ensemble = best_models

    # =========================================================
    # Train a final neural network on the updated labeled set
    # =========================================================
    for i, model in enumerate(ensemble):
        print("Final training of ensemble model: ", model.name, " ...")

        # Compute the steps per epoch and validation steps
        steps_per_epoch = int(len(y_labeled) / batch_size)
        test_steps = int(len(y_test) / batch_size)

        # reset the learning rate before training
        K.set_value(model.optimizer.learning_rate, 0.001)
        reduced_lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.1,
            patience=10,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        )

        # check if the ensemble model files exists and load them up
        check_pt_file = modelsFolder + "/" + model.name + "_" + str(ratio_noisy) + "_Final.h5"
        if os.path.isfile(check_pt_file):
            print("Loading the model weight file...")
            model.load_weights(check_pt_file)

        model_checkpoint = ModelCheckpoint(
            check_pt_file,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        # early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)

        history = model.fit(
            X_labeled, y_labeled,
            callbacks=[model_checkpoint, reduced_lr],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            validation_steps=test_steps
        )


if __name__ == "__main__":
    # ===================
    # Hyperparameters
    # ===================
    confidence_delta = 0.4  # 0.07
    co_train_iter = 5
    bootstrap_iter = 5
    epochs = 250
    ratio_noisy = [0.8, 0.7, 0.6, 0.5]  # ratio of data to be made noisy
    subset_ratio = 0.5  # subset of data to be used in semi-supervised learning
    it_delta = 0.02  # 0.02 is 2%. If the unlabelled data is less than 2% of the overall data, stop training.

    # ==============================
    # Dataset Specific parameters
    # =============================
    batch_size = 1024
    norm = "skin"
    dataset_folder = "Datasets/SkinSegmentationRGB"

    # batch_size = 64
    # norm = None
    # dataset_folder = "Datasets/DryBeanDatasetNormalised"

    # batch_size = 1024
    # norm = "radar"
    # dataset_folder = "Datasets/RadDataset"

    train_file = dataset_folder + "/train/train.csv"
    test_file = dataset_folder + "/test/test.csv"
    modelsFolder = dataset_folder + "/models"

    # Train for each noisy set (large noise to smaller noise~)
    for n_ratio in ratio_noisy:
        # ==============================
        # Prepare the dataset
        # ==============================
        train_dataset_semi_sup, train_dataset_baseline, test_dataset = \
            prepare_dataset(train_file, test_file, n_ratio, norm)

        train_samples = len(train_dataset_semi_sup[0]) + len(train_dataset_semi_sup[2])
        test_samples = len(test_dataset[0])

        # ==================================
        # save the hyperparameters
        # =================================
        parameters = {
            "dataset_folder": dataset_folder,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "confidence_delta": confidence_delta,
            "co_train_iter": co_train_iter,
            "bootstrap_iter": bootstrap_iter,
            "epochs": epochs,
            "ratio_noisy": n_ratio,
            "subset_ratio": subset_ratio,
            "batch_size": batch_size,
            "train_dataset_semi_sup": train_dataset_semi_sup,
            "train_dataset_baseline": train_dataset_baseline,
            "test_dataset": test_dataset,
            "modelsFolder": modelsFolder,
            "it_delta": it_delta
        }
        # parameters_to_file(parameters, dataset_folder)

        # =============================================
        # Get the baseline and semi-supervised datasets
        # =============================================

        # 1. Train the baseline model
        #if not os.path.exists(modelsFolder+"/weights_" + str(n_ratio) + ".h5"):
        train_baseline(parameters)
        # else:
            # print("Baseline Model already trained on noisy dataset...")

        # TODO: Uncomment this
        # 2. Train the semi-supervised
        # train_semi_supervised_ensemble(parameters)