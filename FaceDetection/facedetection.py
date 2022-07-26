import cv2

alg = "haarCascadeFrontalFaceAlgorithm.xml" #xml file with algo
haar_casade = cv2.CascadeClassifier(alg) #loading the model with cv2 library

cam = cv2.VideoCapture(0) #initializing the camera

while True:
    _, img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_casade.detectMultiScale(grayImg, 1.3, 4)
    text = "No Person Detected"
    for (x,y,w,h) in face:
        text = "Person Detected"
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        
    cv2.putText(img,text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    cv2.imshow("Face Detection Using Haar Cascade",img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()