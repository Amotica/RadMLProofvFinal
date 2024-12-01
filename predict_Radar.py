from dataset_helpers import *
import nexradaws
import numpy as np
import os, csv
from models import *
from PIL import Image
from tqdm import tqdm
import pyart
import matplotlib.pyplot as plt

#conn = nexradaws.NexradAwsInterface()


def SaveRadarData(predict_folder):
    for scan_name in os.listdir(predict_folder):
        scan_path = os.path.join(predict_folder, scan_name)

        output_folder = os.path.join(predict_folder, os.path.splitext(scan_name)[0])
        os.makedirs(output_folder, exist_ok=True)

        print("Processing: ", output_folder, "...")

        radar = pyart.io.read_nexrad_archive(scan_path)
        # radar = pyart.io.read_cfradial(scan_path)
        display = pyart.graph.RadarDisplay(radar)

        # Define properties for each plot
        plots = [
            {
                'field': 'reflectivity',
                'vmin': -30.0,
                'vmax': 75.0,
                'cmap': 'pyart_NWSRef',
                'title': 'Horizontal Reflectivity',
                'filename': 'horizontal_reflectivity.png',
                'axislabels': ('', 'North South distance from radar (km)')
            },
            {
                'field': 'differential_reflectivity',
                'vmin': -1.0,
                'vmax': 8.0,
                'cmap': 'pyart_RefDiff',
                'title': 'Differential Reflectivity',
                'filename': 'differential_reflectivity.png',
                'axislabels': ('', '')
            },
            {
                'field': 'differential_phase',
                'vmin': -180.0,
                'vmax': 180.0,
                'cmap': 'pyart_Wild25',
                'title': 'Differential Phase',
                'filename': 'differential_phase.png',
                'axislabels': ('', '')
            },
            {
                'field': 'cross_correlation_ratio',
                'vmin': 0.0,
                'vmax': 1.05,
                'cmap': 'pyart_RefDiff',
                'title': 'Correlation Coefficient',
                'filename': 'correlation_coefficient.png',
                'axislabels': ('East West distance from radar (km)', '')
            }
        ]

        # Create and save each plot separately
        for plot in plots:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Manually set axes
            display.plot(plot['field'], 0, ax=ax,
                         title=plot['title'], cmap=plot['cmap'],
                         vmin=plot['vmin'], vmax=plot['vmax'],
                         axislabels=plot['axislabels'])
            display.set_limits((-150, 150), (-150, 150), ax=ax)
            plt.savefig(os.path.join(output_folder, plot['filename']))
            plt.close(fig)  # Close each plot after saving


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

            return features, unique_classes.shape[0]

    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None, None

def reconstruct_radar(Z, V, SW, ZDR, PHV, DP, Y, radar):
    fields = {}

    # Reflectivity
    fields['reflectivity'] = {
        'data': Z,
        'units': 'dBZ',
        'long_name': 'reflectivity'
    }

    # Velocity
    fields['velocity'] = {
        'data': V,
        'units': 'm/s',
        'long_name': 'velocity'
    }

    # Spectrum Width
    fields['spectrum_width'] = {
        'data': SW,
        'units': 'm/s',
        'long_name': 'spectrum_width'
    }

    # Differential Reflectivity
    fields['differential_reflectivity'] = {
        'data': ZDR,
        'units': 'dB',
        'long_name': 'differential_reflectivity'
    }

    # Cross Correlation Ratio
    fields['cross_correlation_ratio'] = {
        'data': PHV,
        'units': '',
        'long_name': 'cross_correlation_ratio'
    }

    # Differential Phase
    fields['differential_phase'] = {
        'data': DP,
        'units': 'degrees',
        'long_name': 'differential_phase'
    }

    # Predictions
    fields['predictions'] = {
        'data': Y,
        'units': '',
        'long_name': 'predictions'
    }

    # Create the radar object with only the required fields and radar geometry for plotting
    new_radar = pyart.core.Radar(
        fields=fields,
        time=radar.time,
        metadata={},
        scan_type=radar.scan_type,  # Placeholder for scan type (you can replace with actual value)
        latitude=radar.latitude,  # Copy from the original radar object
        longitude=radar.longitude,  # Copy from the original radar object
        altitude=radar.altitude,  # Copy from the original radar object
        sweep_number=radar.sweep_number,  # Copy from the original radar object
        sweep_mode=radar.sweep_mode,  # Copy from the original radar object
        fixed_angle=radar.fixed_angle,  # Copy from the original radar object
        sweep_start_ray_index=radar.sweep_start_ray_index,  # Copy from original
        sweep_end_ray_index=radar.sweep_end_ray_index,  # Copy from original
        elevation=radar.elevation,  # Copy from original radar object
        azimuth={'data': radar.azimuth['data']},  # Add azimuth data
        _range={'data': radar.range['data']}  # Add range data
    )

    # You can now plot using Py-ART's plotting functions
    pyart.graph.RadarDisplay(new_radar).plot('predictions', 0)  # Example plot (Reflectivity, sweep 0)

def read_radar(file):
    # Open the radar data
    radar = pyart.io.read_nexrad_archive(file)
    # Get specified field from the scan
    sweep0 = radar.get_slice(0)

    # Get specified field from the scan
    Z = radar.fields["reflectivity"]["data"][sweep0]  # reflectivity
    V = radar.fields["velocity"]["data"][sweep0]  # velocity
    SW = radar.fields["spectrum_width"]["data"][sweep0]  # spectrum_width
    # Dual polarisation Products
    ZDR = radar.fields["differential_reflectivity"]["data"][sweep0]  # differential_reflectivity
    PHV = radar.fields["cross_correlation_ratio"]["data"][sweep0]  # cross_correlation_ratio
    DP = radar.fields["differential_phase"]["data"][sweep0]  # differential_phase

    sweep_azimuth = radar.azimuth['data'][sweep0]
    sweep_range = radar.range['data']

    return np.stack((Z, V, SW, ZDR, PHV, DP), axis=-1), radar


def predict_Radar(predict_folder, test_file, modelsFolder, suffix="_baseline.h5"):
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
        BasicFNetModel(features, num_classes, testing=True)
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
        # Process radar scans in predict_folder
        # =================================================================
        for scan_name in tqdm(os.listdir(predict_folder), desc=f"Predicting with {model.name}"):
            scan_path = os.path.join(predict_folder, scan_name)
            #print(scan_path)

            # Load the scan as 6 channels
            try:
                # TODO: read the .gz file and stack the products as channels
                X, radar = read_radar(scan_path)
                # print(radar.scan_type)
                # SaveRadarData(scan_path, predict_folder, scan_name)
            except Exception as e:
                print(f"Error loading image {scan_path}: {e}. Skipping.")
                continue

            # Normalize the data appropriately
            # ================================================================
            # Clip the values in range of the radar product min and max values
            # ================================================================
            valid_mask = X != -9999

            # Clip the values only where valid (not -9999) for each channel
            X[:, :, 0] = np.where(valid_mask[:, :, 0], np.clip(X[:, :, 0], -32.0, 94.5), X[:, :, 0])  # Channel 1
            X[:, :, 1] = np.where(valid_mask[:, :, 1], np.clip(X[:, :, 1], -95.0, 95.0), X[:, :, 1])  # Channel 2
            X[:, :, 2] = np.where(valid_mask[:, :, 2], np.clip(X[:, :, 2], -63.5, 63.0), X[:, :, 2])  # Channel 3
            X[:, :, 3] = np.where(valid_mask[:, :, 3], np.clip(X[:, :, 3], -7.875, 7.9375), X[:, :, 3])  # Channel 4
            X[:, :, 4] = np.where(valid_mask[:, :, 4], np.clip(X[:, :, 4], 0.0, 1.0), X[:, :, 4])  # Channel 5
            X[:, :, 5] = np.where(valid_mask[:, :, 5], np.clip(X[:, :, 5], 0.0, 360.0), X[:, :, 5])  # Channel 6

            # ============================
            # Normalise the train data
            # ============================
            X[:, :, 0] = np.where(valid_mask[:, :, 0], (X[:, :, 0] + 32.0) / 126.5, X[:, :, 0])  # Channel 1
            X[:, :, 1] = np.where(valid_mask[:, :, 1], (X[:, :, 1] + 95.0) / 190.0, X[:, :, 1])  # Channel 2
            X[:, :, 2] = np.where(valid_mask[:, :, 2], (X[:, :, 2] + 63.5) / 126.5, X[:, :, 2])  # Channel 3
            X[:, :, 3] = np.where(valid_mask[:, :, 3], (X[:, :, 3] + 7.875) / 15.8125, X[:, :, 3])  # Channel 4
            X[:, :, 4] = np.where(valid_mask[:, :, 4], (X[:, :, 4] + 0.0) / 1.0, X[:, :, 4])  # Channel 5
            X[:, :, 5] = np.where(valid_mask[:, :, 5], (X[:, :, 5] + 0.0) / 360.0, X[:, :, 5])  # Channel 6

            # =================================================================
            # Predict on the dataset
            # =================================================================
            # Flatten the input data into a 2D array (num_pixels, num_channels)
            flat_X = X.reshape(-1, 6)

            # Create a mask for valid pixels (those not equal to -9999)
            valid_mask2 = flat_X != -9999

            # Predict only for valid pixels
            flat_Y_pred = model.predict(flat_X)  # Predict class probabilities for each pixel

            # Convert predicted probabilities to class labels
            flat_Y_pred = np.argmax(flat_Y_pred, axis=1)

            # Reshape back to the image dimensions
            Y_pred = flat_Y_pred.reshape(X.shape[:2])

            # Set predictions to NaN where the original pixel was invalid (i.e., -9999)
            Y_pred = np.where(np.all(valid_mask2, axis=1).reshape(X.shape[:2]), Y_pred, np.nan)

            reconstruct_radar(
                X[:, :, 0],
                X[:, :, 1],
                X[:, :, 2],
                X[:, :, 3],
                X[:, :, 4],
                X[:, :, 5],
                Y_pred,
                radar
            )

            # # Convert polar coordinates (azimuth, range) to Cartesian coordinates for plotting
            # azimuth_rad = np.deg2rad(sweep_azimuth)
            # r, theta = np.meshgrid(sweep_range, azimuth_rad)
            # x = r * np.sin(theta)
            # y = r * np.cos(theta)
            #
            # # Plot the data
            # plt.figure(figsize=(10, 8))
            # plt.pcolormesh(x, y, Y_pred, cmap="pyart_RefDiff", shading="auto")
            # #plt.colorbar(label="Differential Reflectivity (dB)")
            # #plt.title(f"{field_name.capitalize()} - Sweep {sweep_index}")
            # #plt.xlabel("Distance (km)")
            # #plt.ylabel("Distance (km)")
            # plt.axis("equal")  # Ensure the aspect ratio is correct
            # plt.show()
            #
            # # =================================================================
            # # Convert class predictions to RGB
            # # =================================================================
            # # Define a colormap for classes
            # # Updated colormap with an entry for NaN values
            # colormap = {
            #     0: [0, 0, 0],  # Class 0 -> Black
            #     1: [255, 255, 255],  # Class 1 -> White
            #     2: [255, 0, 0],  # Class 2 -> Red
            #     3: [0, 255, 0],  # Class 3 -> Green
            #     4: [0, 0, 255],  # Class 4 -> Blue
            #     5: [255, 255, 0],  # Class 5 -> Yellow
            #     6: [255, 0, 255],  # Class 6 -> Magenta
            #     7: [0, 255, 255],  # Class 7 -> Cyan
            #     'NaN': [128, 128, 128]  # NaN -> Gray (or any color you prefer)
            # }
            #
            # # Create an RGB representation of the prediction
            # Y_pred_RGB = np.zeros((*Y_pred.shape, 3), dtype=np.uint8)
            #
            # # Iterate over each class in the colormap
            # for class_id, color in colormap.items():
            #     if class_id == 'NaN':
            #         # Apply NaN color where Y_pred is NaN
            #         Y_pred_RGB[np.isnan(Y_pred)] = color
            #     else:
            #         # Apply class colors
            #         Y_pred_RGB[Y_pred == class_id] = color
            #
            # # =================================================================
            # # Save the RGB predictions
            # # =================================================================
            # new_folder_path = os.path.join(predict_folder, suffix)
            # new_folder_path = os.path.join(new_folder_path, model.name)
            # os.makedirs(new_folder_path, exist_ok=True)
            # output_path = os.path.join(new_folder_path, f"{scan_name}_{model.name}.png")
            # try:
            #     pred_img = Image.fromarray(Y_pred_RGB)  # Convert to PIL Image
            #     pred_img.save(output_path)  # Save the prediction
            #     print(f"Saved prediction to {output_path}.")
            # except Exception as e:
            #     print(f"Error saving prediction for {scan_name}: {e}.")



if __name__ == "__main__":
    # ==============================
    # Dataset Specific parameters
    # =============================
    batch_size = 1024
    norm = "skin"
    dataset_folder = "Datasets/RadDataset"
    predict_folder = dataset_folder + "/predict/bird"
    test_file = dataset_folder + "/test/test.csv"
    modelsFolder = dataset_folder + "/models"
    extract_products = False

    if extract_products:
        SaveRadarData(predict_folder)


    # predict_Radar(
    #     predict_folder,
    #     test_file,
    #     modelsFolder,
    #     suffix="_baseline_0.6.h5"
    # )

    predict_Radar(
        predict_folder,
        test_file,
        modelsFolder,
        suffix="_0.6.h5"
    )

    # predict_Radar(
    #     predict_folder,
    #     test_file,
    #     modelsFolder,
    #     suffix="_0.6_Final.h5"
    # )