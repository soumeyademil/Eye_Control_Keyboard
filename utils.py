from threading import Thread
import cv2


ALPHABET = 'abcdefghijklmnopqrstuvwxyz '


# WebCam Video Stream using threads to increase FPS
class VideoStream:
    '''
    Source : https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    '''
    def __init__(self,src=0) -> None:
        self.stream = cv2.VideoCapture(src)
        _, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            _, self.frame = self.stream.read()

    
    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


if __name__ == "__main__":
    pass
    