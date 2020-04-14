# Face Recognition with dlib and face_recognition
*A facial recognition script built with dlib by Davis King, and face_recognition by Adam Geitgey.*

* Dlib is a C++ toolkit containing machine learning algorithms used for real world problems and tools for creating complex software in C++ to solve real world problems.

* face_recognition is simply a face recognition API for Python.

So, what the script does is ```encode_faces.py``` creates a 128-d vector for the dataset, serializes it, then outputs a pickle file. ```recognize_faces_image.py``` and ```recognize_faces_video.py``` detects the faces from the loaded images and video streams then tries to match it based from the serialized data from the pickle file.
