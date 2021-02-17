import cv2
import os
import numpy as np
import faceRecognition as fr


#This module takes images stored in disk,and performs face recognition
test_img=cv2.imread('C:/Users/Aditya/.spyder-py3/testing/27/SarE_02317_f_22_i_nf_nc_hp_2016_1_e0_nl_o.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)


#Comment belows lines when running this program second time.
#faces,faceID=fr.labels_for_training_data('C:/Users/Aditya/.spyder-py3/trainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')


#Uncomment below line for subsequent runs
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

name={0:"BOY1",1:"GIR1L",2:"BOY2",3:"BOY3",4:"BOY4",5:"BOY5",6:"BOY6",7:"GIRL2",8:"GIRL3",9:"john",10:"lib",11:"mike",12:"pat",13:"sar",14:"ste",15:"stu",16:"tom",17:"will",18:"AbdA",19:"AboA",20:"AlaG",21:"BahA",22:"MahA",23:"MerM",24:"MirM",25:"MohG",26:"OliA",27:"SarE",28:"YosB",29:"Aditya"}
for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,threshold=face_recognizer.predict(roi_gray)
    print("threshold:",threshold)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
#    if(threshold>50):#If threshold more than 50 then don't print predicted face text on screen
#        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(200,200))
cv2.imshow("face detection",resized_img)
cv2.waitKey(0)#Waits until a key is pressed
cv2.destroyAllWindows()
