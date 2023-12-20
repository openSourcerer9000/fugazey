import cv2
# from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

resx,resy = 1920,1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resx )
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resy)

r, frame = cap.read()
frame.shape
cap.release()
xcenter=int(resx/2)
ycenter=660
yoffset = 325
xoffset = 450
while True:
    cv2.imshow('croppd',frame[ycenter-yoffset:ycenter+yoffset,xcenter-xoffset:xcenter+xoffset])
# cv2.imshow('adsf',frame)
