import numpy as np
import cv2
import time



cap = cv2.VideoCapture(1)


while (True):
    ret, back = cap.read()
    cv2.putText(back, "Press y to save background image",(20,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('background.png', back)
        cv2.destroyAllWindows()
        break
    cv2.imshow('Background',back)

img = cv2.imread('background.png')
backround = cv2.resize(img, (640, 480))
backround = cv2.cvtColor(backround, cv2.COLOR_BGR2GRAY)

def boundingbox(frame,backround_img):

    diff = cv2.absdiff(frame, backround_img)      # subtract the frame from the original backround
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Covert to gray scale
    blur = cv2.GaussianBlur(diff, (5, 5), 0.5)    # Gaussian filter
    thresh = cv2.threshold(blur, 50, 200, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1] # Binary Thershold using otsu

    # Applying Morphological transformations
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #Finding the bounderys of the image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI_number = 0
    h_max = 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # ROI = frame[y:y + h, x:x + w]
        # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        if h > h_max: # TODO change to area
            x_max, y_max, w_max, h_max = x, y, w, h
    cv2.rectangle(frame, (x_max - 15, y_max - 15), (x_max + w_max + 30, y_max + h_max + 30), (36, 255, 12), 5)

    return [x_max, y_max, w_max, h_max] , thresh

def ROI(img,BB):
    x,y,w,h = BB
    ROI = img[y:y + h, x:x + w]
    return ROI




cap = cv2.VideoCapture(1)



while True:

    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    # start = time.time()



    # End time
    # end = time.time()
    # calculate the FPS for current frame detection
    # fps = 1 / (end - start)

    # print(f"{fps:.2f} FPS")
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (640, 480))
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    BB, BW_mask = boundingbox(frame1, backround)
    roi = ROI(frame1, BB)
    cv2.imshow('Mask', BW_mask)


    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()