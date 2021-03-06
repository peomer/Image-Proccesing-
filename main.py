import numpy as np
import cv2 as cv2
import time
import turtle
import threading
cv2.startWindowThread()



# cap = cv2.VideoCapture("malca.mp4")
cap = cv2.VideoCapture(1)


#Take backround image :
while (True):
    ret, back = cap.read()
    cv2.putText(back, "Press y to save background image",(20,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('background.png', back)
        cv2.destroyAllWindows()
        break
    cv2.imshow('Background',back)

# cv2.imwrite('images.png', back)
img = cv2.imread('background.png')

backround = cv2.resize(img, (640, 480))
backround = cv2.cvtColor(backround, cv2.COLOR_BGR2GRAY)



def boundingbox(frame,backround_img,users=2,factor=0.1):

    diff = cv2.absdiff(frame, backround_img)      # subtract the frame from the original backround
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Covert to gray scale
    blur = cv2.GaussianBlur(diff, (5, 5), 0.8)    # Gaussian filter
    thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)[1] # Binary Thershold using otsu
    # Applying Morphological transformations
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('BB', thresh)
    # cv2.waitKey(0)
    #Finding the bounderys of the image
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI_number = 0
    h_max = 0
    BBs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bb_area = w*h
        x=int(x-factor*w); y=int(y-factor*h); w=int(w+factor*w); h=int(h+factor*h)
        tup_of_bb = ([x,y,x+w,y+h],bb_area)
        BBs.append(tup_of_bb)
    BBs = sorted(BBs, key=lambda tup: tup[1])
    Top_users_bbs = BBs[-users:]
    BBs = sorted(Top_users_bbs, key=lambda tup: tup[0][0])
    BBs = [elem[0] for elem in BBs]
    return BBs

def ROI(img,BB):
    x,y,w,h = BB
    ROI = img[y:y + h, x:x + w]
    return ROI


def motion_detect(flow, thresh):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    sub_bgr = np.linalg.norm(bgr)
    moved = True
    if sub_bgr < thresh:
        moved = False
    else:
        moved = True

    return moved


Move_time = 2
PLAY = True # while playing the game
STOP = False
i = 2  # frame counter
# Players = input('How many players are playing with us?')

#### Set up Screen
width = 600
height = 600
s = turtle.Screen()
s.setup(width,height)
s.bgcolor("red")
s.title("RED light Green Light")

init_counter = 4
pen= turtle.Pen()
pen.hideturtle()
pen.write("Welcome to Squid games.",align="center",font=(None,30))
time.sleep(1)
pen.clear()
pen.write("The game will begin in...",align="center",font=(None,30))
time.sleep(1)
for timer in range(init_counter,-1,-1):
    pen.clear()
    pen.write(timer,align="center",font=(None,80))
    time.sleep(1)
pen.clear()
pen.color("red")
s.bgcolor("lightgreen")
pen.write("Start the Game ! ",align="center",font=(None,50))
time.sleep(1)
pen.clear()
####
def green_screen():
    go = turtle.Screen()
    go.setup(width, height)
    go.bgcolor("green")
    pen = turtle.Pen()
    pen.hideturtle()
    pen.write("GO", align="center", font=(None, 30))
    time.sleep(3)
    pen.clear()

def red_screen():
    go = turtle.Screen()
    go.setup(width, height)
    go.bgcolor("red")
    pen = turtle.Pen()
    pen.hideturtle()
    pen.write("Stop", align="center", font=(None, 30))
    time.sleep(3)
    pen.clear()

GREEN = True
RED = True

while (PLAY):

    # while(GREEN):
    #     green_screen()
    #     break


    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (640, 480))
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    BB , BW_mask = boundingbox(frame1, backround)
    roi = ROI(frame1, BB)
    # cv2.imshow('Mask', roi)


    # Take previous frame

    ret, prev = cap.read()
    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.resize(gray_prev, (640, 480))
    gray_prev_roi = ROI(gray_prev, BB)
    start_freeze =time.time()
    threading.Thread(target=red_screen()).start()
    # red_screen()
    while(time.time() - start_freeze < 10):
        # Take Next frame
        ret, next = cap.read()
        gray_next= cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.resize(gray_next, (640, 480))
        gray_next_roi = ROI(gray_next,BB)

        # Calculate opticl flow :
        flow = cv2.calcOpticalFlowFarneback(gray_prev_roi, gray_next_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        gray_prev_roi = gray_next_roi
        # gray_prev = gray_next
        Moved = motion_detect(flow,300) #TODO change to area of BB as threshold

        if Moved:
            cv2.putText(next, "Moved!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('game', next)
        # STOP = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    # When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
