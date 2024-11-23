"""
code from professor
modified to make pathnames resolve better but idk they're still messy
"""

import os

from ultralytics import YOLO

# PATH = "/Users/FULL_PATH_GOES_HERE/YOLO/"  # Windows: r'C:\Users\...\YOLOv8'
PATH = os.path.join(os.getcwd(), "YOLO")


def main() -> None:
    """main function"""

    # Load a model
    # Build a new model from scratch - Do not change "yolov8" name as it pulls that from hard disk
    model = YOLO("yolov8.yaml")
    # model = YOLO(path + "/last.pt") # If need to resume
    # model = YOLO(path + "/best50epochs.pt")  # Load a pretrained best model

    # Train and use the model
    model.train(data=os.path.join(PATH, "data_yolo.yaml"), epochs=60, patience=10)
    # patience = Number of epochs to wait without improvement in validation metrics before early stopping the training.
    # device = 'mps Uses Apple M1 and M2 chips for a high-performance way of executing computation and image processing
    # metrics = model.val()  # Evaluate model performance on the validation set

    # Define the path to your folder and load the model
    # Change to your folder path containing .jpg and .mp4 files
    folder_path = os.path.join(PATH, "To_Be_Predicted")

    # Loop through each file in the folder and make detections
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".jpg", ".jpeg", ".mp4")):
            file_path = os.path.join(folder_path, file_name)
            model.predict(source=file_path, save=True)


if __name__ == "__main__":
    main()
