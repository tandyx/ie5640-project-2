"""code from professor"""

import os
import tempfile
import argparse

import matplotlib.pyplot as plt
import numpy as np

# from tensorflow import keras
from keras import layers
from keras.api import Sequential
from keras.api.callbacks import EarlyStopping
from keras.api.preprocessing import image
from PIL import Image
from sklearn.model_selection import train_test_split

import shared

# Load and preprocess images (assuming 'images' is a list of image paths)
# PATH = "/Users/FULL_PATH_GOES_HERE/CNN/"  # Windows: r'C:\Users\...'
IMG_SIZE = (640, 480)  # width by height


def load_images(
    basepath: os.PathLike,
    ispart: bool,
    img_size: tuple[int, int] = IMG_SIZE,
    zippath: os.PathLike = None,
) -> list[np.ndarray]:
    """manipulates images and grayscales them or something like that

    args:
        - basepath: the basepath of images
        - ispart: is this a part?
        - img_size: (width, height) of the image
        - zippath: path to zip containing images \n
    returns:
        - list[np.NDArry]: list of image arrays
    """
    label = int(ispart)
    if not os.path.exists(
        image_dir := os.path.join(basepath, "PATH_TO_IMAGES", f"{label}")
    ):
        os.makedirs(image_dir)
        if zippath:
            shared.unzip(zippath, image_dir, True)
    image_files = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.endswith(("jpg", "jpeg", "JPG"))
    ]
    grayscale_images = [np.array(Image.open(imgp).convert("L")) for imgp in image_files]
    output_path = os.path.join(basepath, f"grayscale_images_{label}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx, img_array in enumerate(grayscale_images):
        # Save grayscale image as JPG (or desired format)rs(image_dir)
        Image.fromarray(img_array.astype("uint8"), "L").save(
            os.path.join(output_path, f"image_{idx}.jpg")
        )
    return [
        np.array(Image.open(os.path.join(output_path, fname)).resize(img_size))
        for fname in os.listdir(output_path)
    ]


def save_images_from_array(
    images_array: list[np.ndarray], labels: list[int], output_folder: os.PathLike
) -> None:
    """
    Extra work to reconstruct images if needed.

    saves an image from a list of numpy arrays

    args:
        - images_arrary (list[np.ndarray])
        - labels (list[int])
        - output_folder (os.Pathlike) \n
    returns:
        - None
    """
    for idx, img_array in enumerate(images_array):
        # Assuming the image_array contains the pixel values for the image
        # Create PIL image from array
        img = Image.fromarray(img_array.astype("uint8"))
        # img = img.resize((original_width, original_height))
        # label =   # Get label for the image
        if not os.path.exists(
            label_dir := os.path.join(output_folder, str(labels[idx]))
        ):
            os.makedirs(label_dir)
        img.save(os.path.join(label_dir, f"image_{idx}.jpg"))  # Save the image


def main(**kwargs):

    if not os.path.exists(basepath := os.path.join(tempfile.gettempdir(), "CNN")):
        os.mkdir(basepath)
    # Load images and labels for each class
    product_images = load_images(basepath, True, zippath=kwargs["parts_zip"])
    no_product_images = load_images(basepath, False, zippath=kwargs["no_parts_zip"])

    # Combine the two sets of images and labels
    all_images = product_images + no_product_images
    all_labels = [1] * len(product_images) + [0] * len(no_product_images)

    # Split the dataset into training and testing sets
    x_train, x_valid, y_train, y_valid = [
        np.array(_s)
        for _s in train_test_split(
            all_images, all_labels, test_size=0.4, random_state=1
        )
    ]
    # Save training images
    save_images_from_array(
        x_train, y_train, os.path.join(basepath, "train_output_folder")
    )
    # Save validation images
    save_images_from_array(
        x_valid, y_valid, os.path.join(basepath, "valid_output_folder")
    )

    # *****************************
    # Convolutional Neural Network
    # *****************************
    # image_size = (original_height, original_width)

    model = Sequential(
        [
            layers.Input(
                shape=(*IMG_SIZE[::-1], 1)
            ),  # Update the input shape to reflect grayscale images
            layers.Conv2D(
                32, (3, 3), activation="relu"
            ),  # This is a convolutional layer with 32 filters/kernels of size
            # 3x3. It uses ReLU (Rectified Linear Unit) activation function, which introduces non-linearity into the network.
            layers.MaxPooling2D((2, 2)),
            # Reduces the spatial dimensions (width and height) of the previous layer's output by taking the maximum value in a 2x2 grid
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            # Flattens the 2D output from the previous layer into a 1D array, preparing it to connect to a fully connected layer
            layers.Dense(
                128, activation="relu"
            ),  # A dense (fully connected) layer with 128 neurons and ReLU activation
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # optimizer: Determines how the model will be updated based on the computed gradient and loss value. 'adam' is one of
    # the popular optimizers, known for its adaptive learning rates.

    # loss: Quantifies the difference between the predicted values and the actual target values. 'binary_crossentropy' is
    # often used for binary classification problems.

    # metrics: Are additional metrics used to evaluate the model's performance during training. 'accuracy' is a common
    # metric for classification problems, measuring the ratio of correctly predicted observations to the total observations.

    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=6,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stopping],
    )
    # epochs: Indicates how many times the entire dataset is passed forward and backward through the neural network.
    # Each epoch includes processing all the batches within the dataset.
    # The number of epochs affects how many times the model is exposed to the entire dataset

    # batch_size: Determines the number of samples that are processed before the model is updated. Smaller batch sizes
    # might converge faster but could lead to noisier gradients, while larger batch sizes might provide a smoother
    # gradient but can be computationally expensive.

    val_loss, val_accuracy = model.evaluate(x_valid, y_valid)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    target =kwargs["target"] if os.path.isdir(kwargs["target"]) else [kwargs["target"]]
    for _img_path in os.listdir():        
        # Load and preprocess the image
        img = image.load_img(
            os.path.join(basepath, "grayscale_images_1", "image_0.jpg"),
            color_mode="grayscale",
            target_size=IMG_SIZE[::-1],
        )
        # Replace "1.JPG" with your image of choice
        # Normalize pixel values
        img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
        # Make predictions using the model
        predictions = model.predict(img_array)

        # Assuming predictions are [class_prob]
        # Adjust this threshold as needed
        if predictions[0] > kwargs["threshold"]:
            # Visualization with matplotlib assuming predictions are above threshold
            plt.imshow(img)
            plt.title("Detected Object")
            plt.show()
        else:
            print("No object detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-parts-zip", default="photos_no_part.zip", required=False, help="path to no parts zip file -- zip containing images without any parts"
    )
    parser.add_argument(
        "--parts-zip", default="photos.zip", required=False, help="path to no images with parts zip"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, required=False, help="treshold detection"
    )
    parser.add_argument(
        "--target", "-t", default=None, help="target image(s)"
    )
    
    main(**parser.parse_args().__dict__)
