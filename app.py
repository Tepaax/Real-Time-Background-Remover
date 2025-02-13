import cv2
import pyvirtualcam
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# connecting the internal camera (first camera index will be 0, it is the default)
cap = cv2.VideoCapture(1) # 1 seems to work with mobile

# extracting the camera capture size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))

# loading and resizing the background image
#background_image = cv2.resize(cv2.imread("bg_image.jpeg"), (width, height)) 

# creating segmentation instance for taking the foreground (the person).
segmentor = SelfiSegmentation()

# Added a this for it to show up on OBS as a Video Capture device
with pyvirtualcam.Camera(width, height, fps=60, fmt=pyvirtualcam.PixelFormat.BGR) as cam:

    while True:
        # Reading the captured images from the camera
        ret, frame = cap.read()

        # segmenting the image
        green_bg = (0, 255, 0)
        cutThreshold = 0.4 # higher = more cut, lower = less cut
        segmentated_img = segmentor.removeBG(frame, green_bg, cutThreshold)

        cv2.imshow("Camera", segmentated_img)

        cam.send(segmentated_img)
        cam.sleep_until_next_frame()

        # ending condition
        if cv2.waitKey(1) == ord('q'):
            break

# relasing the sources
cap.release()
cv2.destroyAllWindows()
