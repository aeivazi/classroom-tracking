# Face detection and tracking

The repository contains tools for face detection, face alignment and face tracking.

Face detection is written in cpp by employing [MTCNN implementation](https://github.com/happynear/MTCNN_face_detection_alignment/tree/master/code/codes/vs) that has BSD lisence.

The face alignment and face tracking is written in python and heavily uses openface library [openface library](https://cmusatyalab.github.io/openface/) and 
[dlib](http://dlib.net/) both are Apache License 2.0.


## Face recognition

Face recognition task is done by applying Multi-task Cascaded Convolutional Neural Networks (MTCNN). The current version is tested and build on windows, x64, VC 2015. 
The implementation is heavily built on the this [MTCNN implementation] (https://github.com/happynear/MTCNN_face_detection_alignment/tree/master/code/codes/vs).

To run exe there are number of dependencies required, they can be found in the root of built folder: ./cpp/x64/Release/
Remember that dependencies are for Windows, x64 and VC2015.

To compile you will also need number of external libraries, as discussed [here](https://github.com/happynear/caffe-windows).

This version was tested on CPU only and caffee.binding.dll is built with no CUDA support. Running it on CUDA should be quite straightforward, but not tested yet.


The tool can take in video or folder with split already frames.

The output is an xml that describes face boxes and five landmarks (eyes, nose tip and mouth corners). Xml have to be located in the same folder where all frame images are, as further processing assuming relative path for images. 


### Run face recognition on input video

```
mtcnn.exe -i "c:\classroom-project\test\input.mp4" -o "c:\classroom-project\test\input\faces.xml" -n test_dataset -c "dataset test"
```

### Run face recognition on input directory with split frames

```
mtcnn.exe -i "c:\classroom-project\test\input" -o "c:\classroom-project\test\input\faces.xml" -n test_dataset -c "running test dataset"
```

## Train SVM model for face recoginition

### Align and rescale faces to standard size

Align faces by employing MTCNN landmarks and rescale them to standardize size.

Aligning face is an important step to improve accuracy, as it transform faces to face-front positions and thus reducing
additional variation. It also translate the face so that eyes are located in the same place in the output image for all images.
Aligning is done by means of affine transformation based on face landmarks(eyes).

For training and future prediction images are are used of standard size. Right now default size is 96x96 as the openface trained model is expected such.
If the model will be translated in house any other default size can be used. Check tool --help how to set it. 

The faces are saved as a separate images.

```
PYTHONPATH='.' python2 ./src/utils/align_faces.py /data/test/input/faces.xml /data/test/faces-aligned/ --verbose
```

### Create training set

Right now it is done by hand, sorry for this!:) 
The structure of the folders matters. So in the root folder, there have to be subfolders with unique names for participants, and within participant folder all jpg images.

root_folder
    -participant-1 
       image-1.jpg
       ...
       image-n.jpg
    ...
    -participant-n
       image-1.jpg
       ...
       image-n.jpg

### Create model

Calculates faces features from images and SVM model for face recognition. Features and labels are saved as cvs files, model is saved as pickle file.

```
PYTHONPATH='.' python2 ./src/utils/create_model_face_prediction.py  /data/test/faces-for-svm /data/test/model --verbose
```

### Track faces within xml

```
PYTHONPATH='.' python2 ./src/utils/track_faces.py /data/test/input/faces.xml /data/test/input/faces_tracked.xml /data/test/model --verbose
```



## Useful tools for testing

### Visualize predicted labels on the frames
```
PYTHONPATH='.' python2 ./src/utils/visualize_face_tracking.py //data/test/input/faces_tracked.xml /data/test/visualize-tracking --verbose
```
