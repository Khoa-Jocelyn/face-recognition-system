import os
import cv2
import mediapipe as mp
import pandas as pd
import  time
import numpy as np
from _sqlite3 import connect

count_frame = 0
dir_input = 'videos'
dir_output = 'raw_data'
label = "Face-Turn-Right"
cap = cv2.VideoCapture(dir_input + "/" + label + ".avi")
# print("Shape: %d x %d" % (cap.get(3), cap.get(4)))
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
lm_list = []
lms_id = []
for connect in list(mpFaceMesh.FACEMESH_CONTOURS):
    lms_id.extend(list(connect))
lms_id = set(lms_id)
print("Keypoints: ", lms_id)
print("Num Of Keypoints: ", len(lms_id))


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 
    results = model.process(image)  
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return image, results


def draw_landmarks(image, landmarks):
    mpDraw.draw_landmarks(image,
                          landmarks,
                          mpFaceMesh.FACEMESH_CONTOURS,
                          drawSpec,
                          drawSpec)


def extract_landmarks(image, results, draw_lms=True):
    c_lms = []
    if results.multi_face_landmarks:
        for lms in results.multi_face_landmarks:
            c_lm = []
            for id, lm in enumerate(lms.landmark): 
                if id in lms_id:
                    c_lm.append(lm.x)
                    # print("lm.x: ", lm.x)
                    c_lm.append(lm.y)
                    # print("lm.y: ", lm.y)
                    c_lm.append(lm.z)
                    # print("lm.z: ",lm.z)
            c_lms.append(c_lm)
            # print(len(c_lm))
            if draw_lms:
                draw_landmarks(image, lms)
    return c_lm, c_lms


def extract_keypoints(results):
    faceMesh = np.array([[res.x, res.y, res.z] for res in
                     results.multi_face_landmarks.landmark]).flatten() if results.multi_face_landmarks else np.zeros(468 * 3)
    return faceMesh

    
with faceMesh:
    while True:
        ret, frame = cap.read()
        # print(frame.shape)
        # frame=cv2.flip(frame,1)
        if ret:
            prev_frame_time = time.time()
            frame, results = mediapipe_detection(frame, faceMesh)
            if results.multi_face_landmarks is None:
                continue
            c_lm, c_lms = extract_landmarks(frame, results)
            lm_list.append(c_lm)
            fps = 1 / (time.time() - prev_frame_time)
            cv2.putText(frame, "Fps: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.imshow(dir_input + "/" + label + ".avi", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        count_frame += 1
    df = pd.DataFrame(lm_list)
    df.to_csv(dir_output + "/" + label + ".txt")
cap.release()
cv2.destroyAllWindows()
