# Face detection and tracking toolkit

The toolkit is designed to be used for classroom project, thus using expected structure of the input.
The toolkit heavily uses openface library [openface library](https://cmusatyalab.github.io/openface/) and 
[dlib](http://dlib.net/) both are Apache License 2.0.


## Split input video to images, recognize faces, write results to xml

```
mtcnn.exe -i "c:\Users\myaccount\Desktop\classroom-project\birthday\input.mp4" -o "c:\Users\myaccount\Desktop\classroom-project\birthday\input\faces.xml" -n test_dataset -c "birthday dataset test"
```

If you want to work with a split before video, you can give as input the path to the folder:

```
mtcnn.exe -i "c:\anna\crowd_tracking\23100601S1" -o "C:\anna\crowd_tracking\output_xml.xml" -n test_dataset -c "running test dataset"
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
PYTHONPATH='.' python2 ./src/utils/align_faces.py /home/anna/data/birthday/input/faces.xml /home/anna/data/birthday/faces-aligned/ --verbose
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
PYTHONPATH='.' python2 ./src/utils/create_model_face_prediction.py /home/anna/data/birthday/faces-for-svm /home/anna/data/birthday/model --verbose
```

## Track faces within xml

```
PYTHONPATH='.' python2 ./src/utils/track_faces.py /home/anna/data/birthday/input/faces.xml /home/anna/data/birthday/input/faces-out.xml /home/anna/data/birthday/model --verbose
```