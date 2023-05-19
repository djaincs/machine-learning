#face register
import cv2
import pandas as pd
import face_recognition as fr

file_name = 'faces_data.tsv'

vid = cv2.VideoCapture(0)
fd=cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
)

try:
    face_db = pd.read_csv(file_name,index_col=0, sep='\t')
    data ={
        'name':face_db['name'].values.tolist(),
        'encoding':face_db['encoding'].values.tolist(),
    }
except Exception as e:
    print(e)
    data={'name':[], 'encoding':[]}


names=data['name']
enc=data['encoding']

framelimit = 20
frameCount =0
name = input("enter your name:")
while True:
    flag, img = vid.read()
    if flag:
        faces= fd.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),scaleFactor = 1.1, minNeighbors = 5,
                                     minSize =(50,50)
                                    )
        if len(faces)==1:
            x,y,w,h= faces[0]
            img_faces = img[y:y+h, x:x+w, :].copy()
            img_faces= cv2.resize(img_faces,(400,400),interpolation=cv2.INTER_CUBIC)
            face_encoding= fr.face_encodings(img_faces)

            if len(face_encoding)==1:
                enc.append(face_encoding[0].tolist())
                names.append(name)
                frameCount+=1
                print(frameCount)
                cv2.putText(img,str(frameCount),
                                    (50,50),cv2.FONT_HERSHEY_COMPLEX,1.5,
                                    (0,0,255),8)
                if frameCount == framelimit:
                    data={'name':names,'encoding':enc}
                    pd.DataFrame(data).to_csv(file_name,sep='\t')
                    break
        cv2.imshow("preview",img)
        key = cv2.waitKey(1)
        if key == ord('x'):
            break
cv2.destroyAllWindows()
vid.release()