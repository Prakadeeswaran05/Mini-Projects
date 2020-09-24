import cv2

face_cascade=cv2.CascadeClassifier('C:\\Users\\kesav\\Downloads\\face_blur\\haarcascade_frontalface_default.xml')

        
    
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read(0)
    face_img=frame.copy()
    face_rects=face_cascade.detectMultiScale(face_img)
    for (x,y,w,h) in face_rects:
       
        
        img=face_img[y:y+h, x:x+w]
        img= cv2.GaussianBlur(img, (51,51), 0)
        face_img[y:y+h,x:x+w]=img
       
        
    cv2.imshow('blurred face',face_img)
    #cv2.imshow('blur',img)
    
    k=cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

