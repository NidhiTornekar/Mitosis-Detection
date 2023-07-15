# Mitosis-Detection
Web based Mitosis detection on H&E images obtained from MICCAI 2015 International dataset

Dataset used is the MICCAI 2015 International dataset.
Preprocessing was performed on the images and corresponding masks.
Draw the bounding boxes on the pre-processed images.These are the ground truth images and can be added to the static folder.

Train the model using YOLOv5 and save the best.pt weight and add it to the folder.

Flask provides connection between the front-end and the model.

Run using python app.py command and view results in localhost:5000
Deployed using AWS 
