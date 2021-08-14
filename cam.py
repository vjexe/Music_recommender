import cv2
import numpy as np
from keras.models import load_model
import os
import random
from playsound import playsound

# laptop camera
rgb = cv2.VideoCapture(0)

# droidcam android
# rgb = cv2.VideoCapture('http://ipaddress:port/mjpegfeed?640x480')


# pre - trinaed xml file for detecting faces
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def playmusic(emotion):
    path="D:/Project_music/"+emotion+'/'
    os.chdir(path)
    list=os.listdir(path)
    l=len(list)
    n=random.randint(1,l-1)
    print("Playing : "+ list[n])
    playsound(list[n])


# loading saved cnn model
model = load_model('Recsys.h5')
dom_emo = []


# predicting face emotion using saved model
def get_emo(im):
    im = im[np.newaxis, np.newaxis, :, :]
    im_reshaped = im.reshape(1,48,48,1)
    res = model.predict_classes(im_reshaped,verbose=0)
    emo = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    dom_emo.append(emo[res[0]])
   # print("Emotion detected is ", emo[res[0]])
    return emo[res[0]]


# reshaping face image
def recognize_face(im):
    im = cv2.resize(im, (48, 48))
    return get_emo(im)

count =int(15)
while count >=0:
    _, fr = rgb.read()
    flip_fr = cv2.flip(fr,1)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    count-=1
    
    for (x,y,w,h) in faces:
        fc = fr[y:y+h, x:x+w, :]
        gfc = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        out = recognize_face(gfc)
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        flip_fr = cv2.flip(fr,1)
        cv2.putText(flip_fr, out, (30, 30), font, 1, (255, 255, 0), 2)
    
    cv2.imshow('rgb', flip_fr)

    # press esc to close the window
    k = cv2.waitKey(1) & 0xEFFFFF
    if k==27:   
        break
    elif k==-1:
        continue
    else:
        # print k
        continue
cv2.destroyAllWindows()
print("Most dominant emotion is " ,max(dom_emo,key = dom_emo.count))
playmusic(max(dom_emo,key = dom_emo.count))

