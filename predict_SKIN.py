from dataset_helpers import *
import numpy as np
import os, csv
from models import *
from PIL import Image
from tqdm import tqdm


def get_features_and_classesV2(file_path):
    """
    Analyze the CSV file to determine the number of features and classes.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - features: Number of features (columns - 1).
    - classes: Unique values in the last column.
    """
    try:
        # Open the CSV file
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)

            # Read the first row to determine the number of columns
            first_row = next(reader)
            num_columns = len(first_row)

            # Calculate the number of features
            features = num_columns - 1

            # Collect all rows to extract the last column
            last_column = [row[-1] for row in reader]
            last_column.insert(0, first_row[-1])  # Add last column of the first row

            # Convert to NumPy array for unique calculation
            unique_classes = np.unique(last_column)
            #print(unique_classes)

            return features, unique_classes.shape[0]

    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None, None



def predict_RGB(predict_folder, test_file, modelsFolder, suffix="_baseline.h5"):
    """
    Predict the RGB representation of the dataset using an ensemble of models.

    Parameters:
    - predict_folder: Folder containing images to predict.
    - test_dataset: Tuple (x_test, y_test), used for feature extraction and normalization setup.
    - modelsFolder: Path to the folder containing saved model weights.
    - suffix: File suffix for model weight files.
    """
    # =================================================================
    # Create the ensemble of classifiers
    # =================================================================
    features, num_classes = get_features_and_classesV2(test_file)
    ensemble = [
        SNetModelDouble(features, num_classes, testing=True),
        FNetModelDouble(features, num_classes, testing=True),
        #BasicFNetModel(features, num_classes, testing=True)
    ]

    for i, model in enumerate(ensemble):
        print("      >>> Testing Ensemble Model: ", model.name, " ...")

        # =================================================================
        # Load model weights
        # =================================================================
        file_name = model.name + suffix
        check_pt_file = os.path.join(modelsFolder, file_name)

        if os.path.isfile(check_pt_file):
            print("Loading the model weight file " + file_name + "...")
            model.load_weights(check_pt_file)
        else:
            print(f"Model weight file {file_name} not found. Skipping this model.")
            continue

        # =================================================================
        # Process images in predict_folder
        # =================================================================
        for img_name in tqdm(os.listdir(predict_folder), desc=f"Predicting with {model.name}"):
            img_path = os.path.join(predict_folder, img_name)

            # Load the image as RGB and convert to BGR
            try:
                img = Image.open(img_path).convert("RGB")  # Load as RGB
                X = np.array(img)[:, :, ::-1]  # Convert to BGR by reversing the last dimension
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Skipping.")
                continue

            # Normalize the data appropriately
            # norm = np.max(X)  # Example normalization factor
            X_normalized = X / 255.0  # Ensure normalization works with your model
            # test_dataset = prepare_test_dataset(X_normalized, norm)

            # =================================================================
            # Predict on the dataset
            # =================================================================
            flat_X = X_normalized.reshape(-1, 3)  # Flatten the image into (num_pixels, 3)
            flat_Y_pred = model.predict(flat_X)  # Predict class probabilities for each pixel
            flat_Y_pred = np.argmax(flat_Y_pred, axis=1)  # Get predicted classes
            Y_pred = flat_Y_pred.reshape(X.shape[:2])  # Reshape back to image dimensions

            # =================================================================
            # Convert class predictions to RGB
            # =================================================================
            # Define a colormap for classes
            colormap = {
                0: [0, 0, 0],       # Class 0 -> Black
                1: [255, 255, 255],     # Class 1 -> White
                # Add more classes if necessary
            }

            # Create an RGB representation of the prediction
            Y_pred_RGB = np.zeros((*Y_pred.shape, 3), dtype=np.uint8)
            for class_id, color in colormap.items():
                Y_pred_RGB[Y_pred == class_id] = color

            # =================================================================
            # Save the RGB predictions
            # =================================================================
            new_folder_path = os.path.join(predict_folder, suffix)
            new_folder_path = os.path.join(new_folder_path, model.name)
            os.makedirs(new_folder_path, exist_ok=True)
            output_path = os.path.join(new_folder_path, f"{img_name}_{model.name}.png")
            try:
                pred_img = Image.fromarray(Y_pred_RGB)  # Convert to PIL Image
                pred_img.save(output_path)  # Save the prediction
                print(f"Saved prediction to {output_path}.")
            except Exception as e:
                print(f"Error saving prediction for {img_name}: {e}.")



if __name__ == "__main__":
    # ==============================
    # Dataset Specific parameters
    # =============================
    batch_size = 1024
    norm = "skin"
    dataset_folder = "Datasets/SkinSegmentationRGB"
    predict_folder = dataset_folder + "/predict/pascal_faces"
    test_file = dataset_folder + "/test/test.csv"
    modelsFolder = dataset_folder + "/models"

    predict_RGB(
        predict_folder,
        test_file,
        modelsFolder,
        suffix="_baseline_0.6.h5"
    )

    # predict_RGB(
    #     predict_folder,
    #     test_file,
    #     modelsFolder,
    #     suffix="_0.6.h5"
    # )
    #
    # predict_RGB(
    #     predict_folder,
    #     test_file,
    #     modelsFolder,
    #     suffix="_0.6_Final.h5"
    # )