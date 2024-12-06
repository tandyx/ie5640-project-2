"""CNN.py"""

import argparse
import os
import pathlib
import shutil
import tempfile
import time
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.api import Sequential
from keras.api.callbacks import EarlyStopping
from keras.api.preprocessing import image
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = (640, 480)  # width by height

PathLike = str | os.PathLike | pathlib.Path
PathLikeOpt = PathLike | None


def unzip(zippath: PathLike, pathto: PathLikeOpt = None, flatten: bool = False) -> str:
    """unzips a file and returns the path to the file

    args:
        - zippath (PathLike): path to target archive
        - pathto (Pathlike | None): inferred as the zip name strip the ext into /tmp
        - flatten (bool): ignores structure\n
    returns:
        - path to unzipped file as str
    """
    pathto = pathto or os.path.join(
        tempfile.gettempdir(), pathlib.Path(zippath).name.split(".")[0]
    )
    with ZipFile(zippath) as zipf:
        if not flatten:
            zipf.extractall(pathto)
            return pathto
        for mem in zipf.namelist():
            if not (fname := os.path.basename(mem)) or "__MACOSX" in mem:
                continue
            with zipf.open(mem) as src, open(os.path.join(pathto, fname), "wb") as trgt:
                shutil.copyfileobj(src, trgt)
            # copy file (taken from zipfile's extract)
    return pathto


def load_images(
    basepath: PathLike,
    ispart: bool,
    img_size: tuple[int, int] = IMG_SIZE,
    zippath: PathLikeOpt = None,
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
            unzip(zippath, image_dir, flatten=True)
    imgfs = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.endswith(("jpg", "jpeg", "JPG"))
    ]
    if not os.path.exists(
        output_path := os.path.join(basepath, f"grayscale_images_{label}")
    ):
        os.makedirs(output_path)
    for i, img_array in enumerate(np.array(Image.open(p).convert("L")) for p in imgfs):
        Image.fromarray(img_array.astype("uint8"), "L").save(
            os.path.join(output_path, f"image_{i}.jpg")
        )
    return [
        np.array(Image.open(os.path.join(output_path, fname)).resize(img_size))
        for fname in os.listdir(output_path)
    ]


def save_images_from_array(
    images_array: list[np.ndarray], labels: list[int], output_folder: PathLike
) -> list[str]:
    """
    Extra work to reconstruct images if needed.

    saves an image from a list of numpy arrays

    args:
        - images_arrary (list[np.ndarray])
        - labels (list[int])
        - output_folder (Pathlike) \n
    returns:
        - list of paths
    """
    paths: list[str] = []
    for idx, img_array in enumerate(images_array):
        img = Image.fromarray(img_array.astype("uint8"))
        if not os.path.exists(ldir := os.path.join(output_folder, str(labels[idx]))):
            os.makedirs(ldir)
        img.save(_path := os.path.join(ldir, f"image_{idx}.jpg"))  # Save the image
        paths.append(_path)
    return paths


def main(**kwargs) -> None:
    """main function for file"""
    if os.path.abspath(os.curdir) == os.path.abspath(kwargs["basepath"]):
        kwargs["basepath"] = ".images"
    if os.path.exists(kwargs["basepath"]):
        shutil.rmtree(kwargs["basepath"])
    os.mkdir(kwargs["basepath"])
    part_imgs = load_images(kwargs["basepath"], True, zippath=kwargs["parts_zip"])
    nopart_imgs = load_images(kwargs["basepath"], False, zippath=kwargs["no_parts_zip"])
    # Combine the two sets of images and labels
    all_images = part_imgs + nopart_imgs
    all_labels = [1] * len(part_imgs) + [0] * len(nopart_imgs)

    # Split the dataset into training and testing sets
    x_train, x_valid, y_train, y_valid = [
        np.array(_s)
        for _s in train_test_split(
            all_images, all_labels, test_size=0.4, random_state=kwargs["random_state"]
        )
    ]
    # Save training images
    save_images_from_array(
        x_train, y_train, os.path.join(kwargs["basepath"], "train_output_folder")
    )
    # Save validation images
    _img_paths = save_images_from_array(
        x_valid, y_valid, os.path.join(kwargs["basepath"], "valid_output_folder")
    )

    # *****************************
    # Convolutional Neural Network
    # *****************************
    # image_size = (original_height, original_width)

    model = Sequential(
        [
            # Update the input shape to reflect grayscale images
            layers.Input(shape=(*IMG_SIZE[::-1], 1)),
            # This is a convolutional layer with 32 filters/kernels of size
            layers.Conv2D(32, (3, 3), activation="relu"),
            # 3x3. It uses ReLU (Rectified Linear Unit) activation function, which introduces non-linearity into the network.
            layers.MaxPooling2D((2, 2)),
            # Reduces the spatial dimensions (width and height) of the previous layer's output by taking the maximum value in a 2x2 grid
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            # Flattens the 2D output from the previous layer into a 1D array, preparing it to connect to a fully connected layer
            layers.Flatten(),
            # A dense (fully connected) layer with 128 neurons and ReLU activation
            layers.Dense(128, activation="relu"),
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
        epochs=kwargs["epochs"],
        batch_size=kwargs["batch_size"],
        validation_data=(x_valid[:5], y_valid[:5]),
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

    # Load and preprocess the image

    for _imgpth in _img_paths[:5]:
        print("##################################################")
        print(f"Actual: Part{" " if "0" not in _imgpth else " Doesn't "}Exists")
        img_predict(
            image.load_img(_imgpth, color_mode="grayscale", target_size=IMG_SIZE[::-1]),
            model,
            threshold=kwargs["threshold"],
            show_img=kwargs["show_img"],
        )


def img_predict(
    _image: Image, model: Sequential, threshold: float = 0.5, show_img: bool = False
) -> None:
    """predicts an image according to the model

    args:
        - image (Image) pointer to
        - model (Sequential) pointer to
        - threshold (float) prediciton treshold
    """

    # Replace "1.JPG" with your image of choice
    # Normalize pixel values
    img_array = np.expand_dims(image.img_to_array(_image), axis=0) / 255.0
    # Make predictions using the model
    preds = model.predict(img_array)
    print(msg := f"Predction: Part{" " if preds[0] > threshold else " Doesn't "}Exists")
    if not show_img:
        return
    plt.imshow(_image)
    plt.title(msg)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-parts-zip",
        "-npz",
        default="photos_no_part.zip",
        required=False,
        help="path to no parts zip file -- zip containing images without any parts",
    )
    parser.add_argument(
        "--parts-zip",
        "-pz",
        default="photos.zip",
        help="path to no images with parts zip",
    )
    parser.add_argument(
        "--basepath",
        "-b",
        "-o",
        default=os.path.join(tempfile.gettempdir(), "CNN"),
        help="basepath of path, default /tmp",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="treshold detection",
    )
    parser.add_argument(
        "--random-state",
        "--seed",
        "-s",
        type=int,
        default=int(time.time()),
        help="random seed for train/test split",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=10,
        help="training epocs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="training batches",
    )
    parser.add_argument(
        "--show-img",
        "--show-imgage",
        "--show-imgages",
        "-si",
        default=False,
        action="store_true",
        help="show images? interupts script",
    )
    main(**parser.parse_args().__dict__)
