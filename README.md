# Face Recognition with dlib and face_recognition
*A facial recognition script built with dlib by Davis King, and face_recognition by Adam Geitgey.*

* Dlib is a C++ toolkit containing machine learning algorithms used for real world problems and tools for creating complex software in C++ to solve real world problems.

* face_recognition is simply a face recognition API for Python.

So, what the script does is ```encode_faces.py``` creates a 128-d vector for the dataset, serializes it, then outputs a pickle file. ```recognize_faces_image.py``` and ```recognize_faces_video.py``` detects the faces from the loaded images and video streams then tries to find matches based from the pickle file.

Detection method was set on **HOG (Horizontal of Oriented Gradients)** due to technical issues, sacrificing accuracy in the process. It is best to use **CNN (Convolutional Neural Network)** if it is going to be for a full scale project.

I made this on a Windows OS, so I compiled **dlib** myself. Instructions are as follows:
1. Install **CMake**
2. Download **dlib** from https://github.com/davisking/dlib
3. Make sure you have **C++ Build Tools** installed from Visual Studio.
4. Run ```python setup.py install``` from project directory.

*Reference: PyImageSearch, Deep Learning and Facial Recognition with OpenCV, dlib, and face_recognition*
