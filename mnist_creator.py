import cv2
import numpy as np
from PIL import Image
from classifier import classify

# Create a white square image
drawing = False
ix, iy = -1, -1
img = np.ones((512,512,1), np.uint8) * 255

# Function to draw on the image
def draw(event, x, y, flags, param):
    global ix, iy, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 20, (0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 20, (0), -1)

def create_image():
    global img
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)
    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('s'): # press 's' to save the drawn number
            break
    img = cv2.resize(img, (28, 28)) # Resize to 28x28
    cv2.imwrite('drawn_number.jpg', img) # Save as jpg image
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #create_image()
    #Image.open('drawn_number.jpg').show() # Show the drawn number
    classify('img_1.jpg')#''drawn_number.jpg') # Classify the drawn number
