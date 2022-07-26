import cv2
import os

dataset="dataset"
name="sample"#Enter Your name 

path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.makedirs(path)

(width,height)=(130,100)

count=1

alg = "haarCascadeFrontalFaceAlgorithm.xml" #xml file with algo
haar_casade = cv2.CascadeClassifier(alg) #loading the model with cv2 library

cam = cv2.VideoCapture(0) #initializing the camera

while count<31:
    _, img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_casade.detectMultiScale(grayImg, 1.3, 4)

    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    
        faceonly=grayImg[y:y+h,x:x+w]
        resizeimg=cv2.resize(faceonly,(width,height))

        cv2.imwrite(f"{path}/{count}.jpg",resizeimg)
        count+=1

    cv2.imshow("Face detection",img)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()