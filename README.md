# Face detection and tracking toolkit

The toolkit is designed to be used for classroom project, thus using expected structure of the input.
The toolkit heavily uses openface library [openface library](https://cmusatyalab.github.io/openface/) and 
[dlib](http://dlib.net/) both are Apache License 2.0.

## Align
Cuts face to standardize size (normally 96x96) and align face within the square.

For training and future prediction images are are used of standard size. Default size is 96x96 and it will be used so
across tools if not specify otherwise. Check tool --help how to set it. 

Aligning face is an important step to improve accuracy, as it transform faces to face-front positions and thus reducing
additional variation. Aligning is done by means of affine transformation based on face landmarks(eyes and bottom lip).


###Example to run 
From the root of the project.
```
python2 ./src/openface/align.py ~/data/lfw-subset/input ~/data/lfw-subset/output/aligned --verbose
```

## Create model

Create representations:

```
python2 ./src/openface/calculate_representations.py ~/data/lfw-subset/output/aligned ~/data/lfw-subset/output/model --verbose
```

./batch-represent/main.lua -outDir ~/data/lfw-subset/output/model -data ~/data/lfw-subset/output/aligned
./demos/classifier.py train ~/data/lfw-subset/output/model

## Compare faces

./demos/classifier.py infer ~/data/lfw-subset/output/model/classifier.pkl ~/data/lfw-subset/input/Andre_Agassi/Andre_Agassi_0001.jpg
