# python encode_faces.py --dataset dataset --encodings encodings.pickle

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# Argument parsers
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help = "path to input directory of faces and images")
ap.add_argument("-e", "--encodings", required=True,
    help = "path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help = "face detection model to use: either hog or cnn")
args = vars(ap.parse_args())

# Path to input images
print("[INFO] Quantifying images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialize known encodings and names
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] Processing image {}/{}".format(i + 1,
        len(imagePaths)))
    name = imagePath.split(os.path.sep) [-2]

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the x, y coordinates of the boxes corresponding to each face
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    # Compute the facial embeddings for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# Dump the facial encodings and names to disk
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()