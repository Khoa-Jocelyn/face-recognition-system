import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

count = 0
list_frame = []
frame_write = True
label = "Right"
dir_input = "raw_data"
dir_output = "dataset"
mpDraw = mp.solutions.drawing_utils
raw_data_df = pd.read_csv(dir_input + "/" + label + ".txt")
raw_data = []
raw_data.append(raw_data_df.iloc[:, 1:].values)
num_of_timesteps = 1


def draw_landmark_on_image(mpDraw, landmarks, image):
    point = []
    point_list = []
    for i in range(len(landmarks)):
        point.append(landmarks[i])
        if (i + 1) % 4 == 0:
            point_list.append(point)
            point = []
    for lm in point_list:
        h, w, c = image.shape
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        # point.append((cx, cy))
        cv2.circle(image, (cx, cy), 3, (0, 255, 0), cv2.FILLED)  
    return image


frame = np.ones((480, 640, 3), np.uint8) * 255
for lms in raw_data:
    i = 0
    n_sample = len(lms)
    print(n_sample)
    while i < n_sample:
        frame = np.ones((480, 640, 3), np.uint8) * 255
        # ret, frame = cap.read()
        # frame=cv2.flip(frame,1)
        key = cv2.waitKeyEx(1)  
        lm = []
        if key == ord('a'): 
            if i > 0:
                i = i - 1
        elif key == ord('d'): 
            if i < n_sample: 
                i = i + 1
        elif key == ord(' '):
            if i + num_of_timesteps - 1 > len(lms):
                break
            for j in range (num_of_timesteps):
                list_frame.append(lms[i])
                i = i + 1
            count += 1
        elif key == ord('q'):
            break
        if i > n_sample:
            break
        try:
            frame = draw_landmark_on_image(mpDraw, lms[i], frame)
        except Exception as e:
            print(e)
        cv2.putText(frame, str(i), (30, 50), 2, 2, (0, 255, 0), 2);
        cv2.putText(frame, str(count), (30, 100), 2, 2, (0, 0, 0), 2);
        cv2.imshow("image", frame)
if frame_write:
    df = pd.DataFrame(list_frame)
    df.to_csv(dir_output + "/" + label + ".txt")
