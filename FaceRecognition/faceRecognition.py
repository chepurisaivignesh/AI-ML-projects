'''
Fisherface Recognizer.
-->Fisherfaces algorithm extracts principle
   components that separates one individual
   from another. So,now an individual's
   features can't dominate another person's
   features.
-->Fisherface method will be applied to
   generate feature vector of facial image data
   used by system and then to match vector of
   traits of training image with vector
   characteristic of test image using euclidean
   distance formula

########################################################

LBPHFaceRecognizer.
Local Binary Pattern (LBP) isasimple yet very efficient texture operator which labels
the pixels of an image by thresholding the neighborhood of each pixel and considers
the result asabinary number.It doesn't look at image as a whole, but instead tries to find its local structure by
comparing each pixel to its neighboring pixels.

LBPH uses4parameters
  > Radius-to build the circular local binary pattern and represents the radius around
  the central pixel. It is usually set to 1.
  > Neighbors-:The more sample points you include, the higher the computational
   cost. It is usually set to 8.
  > XGrid-the number of cells in the horizontal direction.
  > YGrid-the number of cells in the vertical direction.

################################################3

Workflow of Face-Recognition:
1.Loading face detection algorithm
2.Loading Classifier for face recognition
3.Training classifier for our dataset
4.Reading frame from camera & pre-processing
5.Face detection by its algorithm
6.Predicting face  by loading frame into model
7.Displays recognized class with its accuracy
'''

import cv2, numpy, os
haar_file = 'haarCascadeFrontalFaceAlgorithm.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
dataset = 'dataset'
print('Training....')

(images,labels,names,id) = ([],[],{},0)

for (subdirs, dirs, files) in os.walk(dataset):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dataset, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id = id + 1

(images,labels) = [numpy.array(lis) for lis in [images,labels]]
print(images, labels)
(width,height) = (130,100)

model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)

webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    (_,im) = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        if prediction[1] < 800:
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg",im)
                cnt = 0
    cv2.imshow('FaceRecognition',im)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()