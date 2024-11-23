# ie5640 group project 2

## product detection in the cyber physical lab

### NOTES

1. This is a team assignment. Refer to Canvas (under People > IE5640) to find your teammates.
2. For the report, include a cover page stating the course number, semester, and the name of all team members.
3. Only one of the group members (on behalf of everyone in the group) needs to submit (a) the PDF report and (b) all images and their annotated labels.
4. You may include as many headers/titles as you want, a table of content, and introduction section, and an Appendix.
5. It is fine if you need to include small portion of your code throughout the text for better understanding. However, most, if not all, of the code must be represented at the end of your report under Appendix. You still need to include at least parts of the outputs throughout the main body.
6. Quality, proper interpretation, and organization is of priority not the length of your report.

### 1) SETUP AND MODEL TRAINING (30 pts.)

You will use CNN to determine if a “new image” includes the trained object. CNN can do the object classification but not detection. You will use YOLO (You Look Only Once) for object detection, in a “new image” and “video”. For YOLO, you need images with the target shapes or objects you want to detect, annotated with bounding boxes or labels indicating the location and class of the objects.

1. Each team in the lab should take 50 pictures of the given product. Pictures should represent all aspects and corners of the product in different (lighting, ...) conditions. Your team also needs 10 pictures of the setting but without the given part. Follow the instructor`s directions.
2. In order to determine the target in each picture, you need to annotate them. You can use “labelImg” tool (watch this video: <https://www.youtube.com/watch?v=gRAyOPjQ9_s>) or any other tools of your choice. Annotation allows you to determine bounding box for the object to be detected in an image/video. If you use labelImg and it crashed, see the resolution here: <https://github.com/HumanSignal/labelImg/issues/872>
3. Unzip the provided folder and keep all of its contents in the same folder (and under any directory of your choice on your computer). It is easier to manage if you install an IDE (Integrated Development Environment) app such as PyCharm (this comes with free licensing for students). An IDE brings all programming tools into one environment, which makes programming more convenient. You then need to create and environment, a folder which includes all the necessary files and libraries to run your Python program. Do a research or ask an AI tool to see how to create and environment and make it active. Then, you need to install multiple libraries. Go to the “Terminal” of the IDE app and install the required libraries all at once using the text file, called “required_libraries.txt”, given to you with this command: pip install -r required_libraries.txt. The installation will take a while.

### 2) PREDICTION AND DETECTION (60 pts.)

Your team needs to predict 5 new images using CNN. Following the instructor’s directions, you also need to record a 20 sec video of the part moving on conveyors and detect a part using YOLO algorithm. Report the results from both methods.

### 3) CONCLUSION AND FUTURE WORK (10 pts.)

What would your team conclude for this project and what are the potential areas for future work?
