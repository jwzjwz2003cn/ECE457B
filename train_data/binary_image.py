import numpy as np
import cv2

im = cv2.imread('lowercases2.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

print (thresh)

#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)
samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>40:
        print (cv2.contourArea(cnt))
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>0:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall_t = cv2.resize(roi,(10,10))
            (retval,roismall) = cv2.threshold(roismall_t, 0, 255, 0)
            cv2.imshow('norm',im)
            print(roismall)
            key = cv2.waitKey(0)

            if key == 27:
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                
                samples = np.append(samples,sample,0)


responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('lowercases.data',samples)
