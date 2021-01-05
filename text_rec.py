import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
pytesseract.pytesseract.tesseract_cmd = 'E:\\ipad\\ml\\tesseract.exe'

# for Text detection in Image
img = cv2.imread('C:\\Users\\com\\Desktop\\DSP\\111.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Detecting Characters
hImg, wImg,_ = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    print(b)
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x,hImg-y), (w,hImg-h), (50, 50, 255),2) # will draw a rectangle
    cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)  # will show the character


cv2.imshow('img', img)
cv2.waitKey(0)

# To make Text colorfull
a = 0
for p in range(1, 544):
    a = a + 1
    for q in range(1, 786):
        if np.all(img[p, q]) != True:
            img[p, q] = [a, q, q + a]
cv2.imshow('img', img)
cv2.waitKey(0)



# For real time Text detection
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN
cap=cv2.VideoCapture(0)      # for real time video
cap.set(cv2.CAP_PROP_FPS,16)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open")
cntr=0;
while (1):
    ret,frame=cap.read()
    cntr=cntr+1;
    if((cntr%15)==0):
        imgH,imgW,_ =frame.shape
        x1,y1,w1,h1=0,0,imgH,imgW
        imgchar=pytesseract.image_to_string(frame)
        imgboxes=pytesseract.image_to_boxes(frame)
        for boxes in imgboxes.splitlines():
            boxes=boxes.split(' ')
            x,y,w,h=int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
            cv2.rectangle(frame,(x,imgH-y),(w,imgH-h),(0,0,255),1)
        #cv2.putText(frame,imgchar,(x1+int(w1/50),y1+int(h1/50)+300),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,225),1)
        cv2.putText(frame, imgchar, (x1,y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 225), 2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow('text',frame)
        if cv2.waitKey(2)& 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

# using saved video
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN
cap=cv2.VideoCapture("E:\\RISHIKESH BPMS\\whatsapp img\\Captures\\vd.mp4")    # using saved video
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open")
cntr=0;
while (1):
    ret,frame=cap.read()
    cntr=cntr+1;
    if((cntr%15)==0):
        imgH,imgW,_ =frame.shape
        x1,y1,w1,h1=0,0,imgH,imgW
        imgchar=pytesseract.image_to_string(frame)
        imgboxes=pytesseract.image_to_boxes(frame)
        for boxes in imgboxes.splitlines():
            boxes=boxes.split(' ')
            x,y,w,h=int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
            cv2.rectangle(frame,(x,imgH-y),(w,imgH-h),(0,0,255),1)
        #cv2.putText(frame,imgchar,(x1+int(w1/50),y1+int(h1/50)+300),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,225),1)
        cv2.putText(frame, imgchar, (x1,y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 225), 2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow('text',frame)
        if cv2.waitKey(2)& 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()












