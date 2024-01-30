import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier  
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier("Image_classifier_model.h5","Model/labels.txt")

offset = 20
imgSize = 300
counter = 0

labels = ["A", "B", "C","D","E","F","G","H","Hello","I","I Love You","J","K","L","M","N","No","O","P","Q","R","S","T","U","V","W","X","Y","Yes","Z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        
        
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset : y + h+offset, x-offset : x + w+offset]
        
        imgCropShape = imgCrop.shape
        
        
        aspectRatio = h/w
        
        
        try:
            if(aspectRatio > 1):
                k = imgSize / h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop,(wCal,imgSize))  
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal)/2)
                imgWhite[:,wGap: wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite,draw=False)
                print(prediction, index)
                
            else:
                k = imgSize / w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal)/2)
                imgWhite[hGap: hCal+hGap,:] = imgResize
                prediction, index = classifier.getPrediction(imgWhite,draw=False)
                print(prediction, index)
                
            cv2.putText(imgOutput,labels[index],(x,y-25),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
            
            # Check if imgCrop is not empty and has a valid size before displaying
            if imgCrop.size > 0:
                cv2.imshow("ImageCrop", imgCrop)
                
        except Exception as e:
            print(f"Error in resizing: {e}")
        
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
