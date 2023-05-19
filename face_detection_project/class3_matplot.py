#to detect face and smile and cature image
import cv2
from time import sleep

fd = cv2.CascadeClassifier(
    cv2.data.haarcascades +
        'haarcascade_frontalface_default.xml'
)

sd = cv2.CascadeClassifier(
    cv2.data.haarcascades +
        'haarcascade_smile.xml'
)

vid = cv2.VideoCapture(0)

sequence = 0
captured = False

#to stop when we want just press x otherwise run infinitely
while not captured:
    #to capture frame
    flag,img = vid.read()
    if flag:

        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        x1,y1,w,h = (200,400,200,200)

        faces = fd.detectMultiScale(img_gray,
                                    scaleFactor = 1.1, 
                                    minNeighbors = 5,
                                    minSize = (80,80)) #1.1 - scale factor , minNeighbors = 5
    #original image scale factor = 1 but we are giving 1.1 toh .1 ke factor se zoom kardega, minSize means ki kitne pixels par camera detect kar paega

        for x1,y1,w,h in faces:
            face = img_gray[y1:y1+h, x1:x1+w].copy()
            #agar hum .copy nahi lagaenge toh jo changes hum face me karenge woh img_gray me bhi ho jaenge
            smiles = sd.detectMultiScale(face,1.1,15,minSize=(50,50))

            if len(smiles) == 1: #agar 1 hi smile hai toh
                #to capture image and break
                sequence += 1
                if sequence == 2:
                    captured = cv2.imwrite('selfie.png',img) 
                    break
                # xs,ys,ws,hs = smiles[0]
                # cv2.rectangle(img,
                #                pt1=(x1+xs,y1+ys), pt2=(x1+xs+ws, y1+ys+hs), #x1,y1 isiliye add kiya hai kyuki img jo hai woh full image hai and face me hum particular face ki cropped image le rhe hai toh smile par rectangle laane ke liye hume face ko extend karna padega img ke saath isiliye humne add kiya hai
                #               color=(0,0,255), thickness=3)   

            else:
                sequence = 0             

            cv2.rectangle(img, 
                      pt1 = (x1,y1), pt2 = (x1+w, y1+h),
                        color=(0,255,0), thickness=5)

        cv2.imshow('Preview',img)
        
        key = cv2.waitKey(1) #click image then wait for 1 sec then again click wait...
        #waitKey jab use lete hai ki koi key press karne e liye aap kitna wait karna chahte ho
        if key == ord('x'):
            break
    else:
        break    
    sleep(0.1) #delay of 0.1 s    
vid.release()        #to close the camera after capturing
cv2.destroyAllWindows()