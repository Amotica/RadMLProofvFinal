from pandas import read_csv
import numpy as np

def LoadPreprocessData(file, normalise):
    # Load dataset
    dataset = read_csv(file, dtype=float)
    dataset = dataset.to_numpy()

    # Normalise the weather data
    # TODO: Normalisation should depend on the dataset
    if normalise == "radar":
        dataset = RadarDataNormalisation(dataset)
    elif normalise == "skin":
        dataset = SkinDataNormalisation(dataset)

    return dataset


def RadarDataNormalisation(dataset):
    # ===============================================================
    # NORMALIZE THE ALL PRODUCTS : But remove error values -9999
    # ===============================================================
    dataset = dataset[dataset[:, 0] != -9999]
    dataset = dataset[dataset[:, 1] != -9999]
    dataset = dataset[dataset[:, 2] != -9999]
    dataset = dataset[dataset[:, 3] != -9999]
    dataset = dataset[dataset[:, 4] != -9999]
    dataset = dataset[dataset[:, 5] != -9999]

    # ================================================================
    # Clip the values in range of the radar product min and max values
    # ================================================================
    dataset[:, 0] = np.clip(dataset[:, 0], -32.0, 94.5)
    dataset[:, 1] = np.clip(dataset[:, 1], -95.0, 95.0)
    dataset[:, 2] = np.clip(dataset[:, 2], -63.5, 63.0)
    dataset[:, 3] = np.clip(dataset[:, 3], -7.875, 7.9375)
    dataset[:, 4] = np.clip(dataset[:, 4], 0.0, 1.0)
    dataset[:, 5] = np.clip(dataset[:, 5], 0.0, 360.0)

    # ============================
    # Normalise the train data
    # ============================
    dataset[:, 0] = (dataset[:, 0] + 32.0) / 126.5  # valid_max: 94.5 - valid_min: -32.0
    dataset[:, 1] = (dataset[:, 1] + 95.0) / 190.0  # valid_max: 95.0 - valid_min: -95.0
    dataset[:, 2] = (dataset[:, 2] + 63.5) / 126.5  # valid_max: 63.0- valid_min: -63.5
    dataset[:, 3] = (dataset[:, 3] + 7.875) / 15.8125  # valid_max: 7.9375 - valid_min: -7.875
    dataset[:, 4] = (dataset[:, 4] + 0.0) / 1.0  # valid_max: 1.0 - valid_min: 0.0
    dataset[:, 5] = (dataset[:, 5] + 0.0) / 360.0  # valid_max: 360.0 - valid_min: 0.0

    return dataset


def SkinDataNormalisation(dataset):
    # ============================
    # Normalise the train data
    # ============================
    dataset[:, :-1] = dataset[:, :-1] / 255.0
    return dataset


def get_features_and_classes(X_labeled, y_labeled):
    # x_train = numpy_dataset[:, :-1]
    # y_train = numpy_dataset[:, -1]
    features = X_labeled.shape[-1]
    num_classes = len(np.unique(y_labeled))
    return features, num_classes


def inject_noise(dataset, ratio_noisy=0.5):
    """
    :param dataset: the clean dataset
    :param ratio_noisy: the ratio of the dataset to make noisy (replace labels with -1)
    :return: the ulabeled and labeled datasets
    """
    # Extract features and labels
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    unique_labels = np.unique(labels)

    # Initialize numpy arrays for clean(labeled) and noisy(unlabeled) data
    labeled_data = np.array([]).reshape(0, dataset.shape[1])
    unlabeled_data = np.array([]).reshape(0, dataset.shape[1])

    # Randomly split data by class labels
    for label in unique_labels:
        # Indices with the current label
        indices = np.where(labels == label)[0]
        # Randomly shuffle indices
        np.random.shuffle(indices)
        # find split point based on ratio_noisy
        split_point = int(ratio_noisy * len(indices))
        # Split the data into clean and noisy
        labeled_indices = indices[split_point:]
        unlabeled_indices = indices[:split_point]
        # Append data to clean_data and noisy_data
        labeled_data = np.vstack((labeled_data, dataset[labeled_indices]))
        # Generate noisy labels for unlabeled_data
        noisy_labels = np.random.choice(unique_labels[unique_labels != label], size=len(unlabeled_indices),
                                        replace=True)

        # Set the last column in unlabeled_data to the generated noisy labels
        temp_unlabelled_data = dataset[unlabeled_indices]
        temp_unlabelled_data[:, -1] = noisy_labels
        unlabeled_data = np.vstack((unlabeled_data, temp_unlabelled_data))

    return labeled_data, unlabeled_data


def random_data_subset(X_labeled, y_labeled, subset_ratio):
    """

    :type y_labeled: object
    """
    random_indices = []

    # get the smallest class size
    # min_class_size = min(np.bincount(y_labeled.astype(int)))
    unique_labels, class_counts = np.unique(y_labeled.astype(int), return_counts=True)
    min_class_size = min(class_counts)

    # Set the subset size for each class
    subset_size_per_class = int(subset_ratio * min_class_size)

    for class_label in np.unique(y_labeled):
        # Get indices of samples belonging to the current class
        class_indices = np.where(y_labeled == class_label)[0]

        # Randomly select subset_size_per_class indices for the current class
        class_random_indices = np.random.choice(class_indices, size=subset_size_per_class, replace=True)

        # Append the selected indices to the overall list
        random_indices.extend(class_random_indices)

    # Convert the list to a NumPy array
    random_indices = np.array(random_indices)

    # Use the selected indices to subset X_labeled and y_labeled
    X_subset = X_labeled[random_indices]
    y_subset = y_labeled[random_indices]

    return X_subset, y_subset


def prepare_dataset(train_file, test_file, ratio_noisy, norm):
    # =================================================================
    # Create the dataset for the semi-supervised learning
    # =================================================================
    train = LoadPreprocessData(train_file, normalise=norm)
    test = LoadPreprocessData(test_file, normalise=norm)

    print("train", train.shape)
    print("test", test.shape)

    labeled_train, unlabeled_train = inject_noise(train, ratio_noisy=ratio_noisy)

    X_labeled, y_labeled = labeled_train[:, :-1], labeled_train[:, -1].astype(int)
    X_unlabeled, y_unlabeled = unlabeled_train[:, :-1], unlabeled_train[:, -1].astype(int)
    X_test, y_test = test[:, :-1], test[:, -1].astype(int)

    train_dataset_semi_sup = [X_labeled, y_labeled, X_unlabeled, y_unlabeled]
    test_dataset = [X_test, y_test]
    # =================================================================
    # Create the noisy training data for the baseline
    # =================================================================
    y_labeled = y_labeled[:, np.newaxis]
    labeled_data = np.concatenate((X_labeled, y_labeled), axis=1)
    y_unlabeled = y_unlabeled[:, np.newaxis]
    unlabeled_data = np.concatenate((X_unlabeled, y_unlabeled), axis=1)
    train_dataset_baseline = np.vstack((unlabeled_data, labeled_data))

    return train_dataset_semi_sup, train_dataset_baseline, test_dataset


def prepare_test_dataset(test_file, norm):
    # ================================
    # Create the dataset for testing
    # ================================
    test = LoadPreprocessData(test_file, normalise=norm)
    print("test", test.shape)

    X_test, y_test = test[:, :-1], test[:, -1].astype(int)
    test_dataset = [X_test, y_test]

    return test_dataset