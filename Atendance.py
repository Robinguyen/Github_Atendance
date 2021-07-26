import os # module he dieu hanh 
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import csv
path = 'Python/Image_Antendance'
images = []
classname = []
myList = os.listdir(path) # tao thu muc path
print(myList) # hien thi ra danh sach
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classname.append(os.path.splitext(cl)[0])
print(classname)

def findEncoding(images):
    encodelist = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def maskAntendance(name):
    with open('Python/Du_lieu.csv','r+') as f:
        myDatalist = f.readlines()
        nameList = []

        for line in myDatalist:
           entry = line.split(',')
           nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodelistknown = findEncoding(images)

print('Encoding Complete')
#su dung camera dung openCv
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None, fx=0.25, fy=0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facecurFame = face_recognition.face_locations(imgs)
    EncodecurFame = face_recognition.face_encodings(imgs,facecurFame)
    for encodeface, faceloc in zip(EncodecurFame, facecurFame):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        faceDis = face_recognition.face_distance(encodelistknown,encodeface)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classname[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+1, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            maskAntendance(name)
    
    cv2.imshow('webcam', img)
    cv2.waitKey(1)      
            



