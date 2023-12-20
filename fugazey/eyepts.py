import cv2
import mediapipe as mp
from pathlib import Path
import pandas as pd, numpy as np
import pyautogui
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camx,camy = 1920,1080
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camx )
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

xcenter=int(camx/2)
ycenter=660
yoffset = 325
xoffset = 450

resx,resy = pyautogui.size()
screen_w, screen_h = resx,resy

dt = 0

pth = Path.cwd()/'data'
pth.mkdir(parents=True,exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        # pupil = np.mean( [np.array([landmark.x,landmark.y]) for landmark in landmarks[474:478] ],axis=0)
        # L = 362
        pupl = 473
        L = 398
        R = 263
        lt,rt,pupil = [np.array([landmarks[idx].x,landmarks[idx].y]) for landmark in (L,R,pupl) ]
        og = np.mean((lt,rt), axis=0)
        w = rt[0] - lt[0]
        h = w*resy/resx # your eye given same aspect ratio of screen
        # posnorm between -1:1 for x and y if you could move your eye all the way to the edge
        posnorm = pupil - og
        posnorm[0] = posnorm[0]/(2*w)
        posnorm[1] = posnorm[1]/(2*h)

        # print(pupil)
        # x,y = pupil
        # x = int(x * frame_w)
        # y = int(y * frame_h)
        # print(x,y)
        # cv2.circle(frame,  (x, y), 3, (0,255, 0))
        for idx in (L,R,pupl):
            landmark = landmarks[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            # if id == 1:
            #     screen_x = screen_w * landmark.x
            #     screen_y = screen_h * landmark.y
                # print(landmark.z)
                # pyautogui.moveTo(screen_x, screen_y)
        # left = [landmarks[145], landmarks[159]]
        # for landmark in left:
        #     x = int(landmark.x * frame_w)
        #     y = int(landmark.y * frame_h)
        #     cv2.circle(frame, (x, y), 3, (0, 255, 255))
        # if (left[0].y - left[1].y) < 0.004:
        #     pyautogui.click()
        #     pyautogui.sleep(1)
    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)