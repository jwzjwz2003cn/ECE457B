import numpy as np
import cv2
import sys

class OCR:
	def __init__(self):
		self.dim = 10
		self.dimrow = self.dim * self.dim
		self.ann = cv2.KNearest()
		

		
		
	def trainImage(self, training_image):	
		im = cv2.imread('{0}.png'.format(training_image))
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5,5), 0)
		thresh = cv2.adaptiveThreshold(blur, 255,1,1,11,2)
		contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		samples =  np.empty((0,self.dimrow))
		targets = []
		ascii = [i for i in range(32,126)]
		for contour in contours:
			if cv2.contourArea(contour)>40:
				print (cv2.contourArea(contour))
				[x,y,w,h] = cv2.boundingRect(contour)
				if  h>15:
					cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
					char = thresh[y:y+h,x:x+w]
					char_norm_t = cv2.resize(char,(self.dim,self.dim))
					(retval,char_norm) = cv2.threshold(char_norm_t, 0, 255, 0)
					cv2.namedWindow('train', cv2.CV_WINDOW_AUTOSIZE)
					cv2.imshow('train',im)
					print(char_norm)
					key = cv2.waitKey(0)
					if key == 27:
						cv2.destroyWindow('train')
						sys.exit()
						
					elif key in ascii:
						print(key)
						targets.append(key)
						sample = char_norm.reshape((1,self.dimrow))
                		samples = np.append(samples,sample,0)
                		
		targets = np.array(targets,np.float32)
		targets = targets.reshape((targets.size,1))
		samples = np.array(samples, np.float32)
		print "training complete"
		cv2.destroyWindow('train')
		np.savetxt('{0}.data'.format(training_image),samples)
		np.savetxt('{0}_targets'.format(training_image),targets)
		
		
		self.ann.train(samples, targets)
		
		
	def ocrImage(self, test_image):
		#im = cv2.imread('{0}.png'.format(test_image))
		#out = np.zeros(im.shape, np.uint8)
		#gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		#blur = cv2.GaussianBlur(gray, (5,5), 0)
		#thresh = cv2.adaptiveThreshold(blur, 255,1,1,11,2)

		#contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		pass



