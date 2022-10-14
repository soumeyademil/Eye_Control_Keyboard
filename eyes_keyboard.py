from time import sleep
from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
import imutils
import dlib
import cv2
from utils import *
import pyautogui
import subprocess as sp
import winsound


# References :
# https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
# https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# https://datatofish.com/control-keyboard-python/



# Download 'shape_predictor_68_face_landmarks.dat' model from :
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2



def face_regions_display(image):

	'''
	Displays detected faces landmarks on the original image using dlib and opencv
	'''

	# Create detector and predictor objects
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	# Resize input, and convert it to grayscale
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Detect faces in the grayscale image
	rects = detector(gray, 1)
	for (_, rect) in enumerate(rects):
		# Get landmarks from face
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# For each region, get name and landmarks coords using FACIAL_LANDMARKS_IDXS dict
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			clone = image.copy()
			# Display region name
			cv2.putText(clone, name, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

			for (x, y) in shape[i:j]:
				# Dispaly landmarks
				cv2.circle(clone, (x, y), 2, (0, 0, 255), -1)			
			cv2.imshow('Image', clone)
			cv2.waitKey(0)

		# Get ROI
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)




def eyes_detection(image):
	'''
	Returns only eyes landmarks vectors
	'''

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	# Process only 1 face
	rect = rects[0]
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# Get only eyes region between 36 and 48 (see documentation)
	i, j = 36, 48
	return shape[i:j]



def eye_aspect_ratio(eye):
	'''
	Returns the eye aspect ratio (see reference #2 above)
	'''

	# Compute the euclidean distance between the vertical landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Compute the euclidean distance between the horizontal landmarks
	C = dist.euclidean(eye[0], eye[3])
	# Compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear


def closed_eyes(eyes_shape, thresh=0.3):
	'''
	Returns True if eyes on the input image are closed, False otherwise
	'''

	# Eyes Landmarks are between 36 and 48
	right_eye = eyes_shape[:len(eyes_shape)//2]
	left_eye = eyes_shape[len(eyes_shape)//2:]
	# Get Total ear
	ear = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2
	# Compare to threshold
	if ear < thresh:
		return True
	else:
		return False

frame = np.array((0, ))



if __name__ == "__main__":
	# For beep sound
	frequency = 500  # Set Frequency To 2500 Hertz
	duration = 1000  # Set Duration To 1000 ms == 1 second

	# Camera ID
	cam = 0
	# Start stream reader thread
	vs = VideoStream(src=cam).start()
	s = ALPHABET # From utils
	i = 0 # To loop over alphabets
	# Eye aspect ratio threshold
	thresh = 0.25

	try:
		notepad = "notepad.exe"
		file_name = "file.txt"
		# Open text file to write
		sub_process = sp.Popen([notepad, file_name])
		while True:
			# Show keyboard shot
			image = cv2.imread(f'KeyBoard_Shots//{i+1}.png', 1)
			cv2.imshow('Image', image)	
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			sleep(2)
			# Get current frame
			frame = vs.read()
			# Detect eyes
			eyes_shape = eyes_detection(frame)
			# Check if caracter selected
			if closed_eyes(eyes_shape, thresh):
				# print(s[i])
				winsound.Beep(frequency, duration)
				pyautogui.write(s[i]) # , interval = 0.5)
			i += 1
			if i == len(s):
				i = 0
	finally:
		vs.stop()
		sub_process.terminate()

