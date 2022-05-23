import cv2
import  time
import numpy as np
import mediapipe as mp
import pickle
import random
import json

classdir = "class/"
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))


def load_model(classdir):
    with open(classdir + 'classifier.pkl', 'rb') as infile:
        (model, class_names) = pickle.load(infile, encoding='latin1')
    return model, class_names


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 
    results = model.process(image)  
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return image, results


def extract_landmarks(image, results, draw_lms=False):
    lms_id = []
    for connect in list(mpFaceMesh.FACEMESH_CONTOURS):
        lms_id.extend(list(connect))
    lms_id = set(lms_id)
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
            if draw_lms:
                mpDraw.draw_landmarks(image,
                          lms,
                          mpFaceMesh.FACEMESH_CONTOURS,
                          drawSpec,
                          drawSpec)
    return c_lm, id


def draw_class_on_image(label, image, *args):
    cv2.putText(image, label, *args, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

    
def predict(image, model):
    image, results = mediapipe_detection(image, faceMesh)
    if results.multi_face_landmarks:
        c_lm, _ = extract_landmarks(image, results)
        c_lm = np.array(c_lm).reshape(1, -1)
        class_name = model.predict(c_lm)
        accuracy = model.score(c_lm, class_name)
        print(json.dumps({"Class": class_name[0], "Accuracy": "%.2f" % (accuracy)}, indent=4), "\n")
        if class_name != ["Normal"]:
            image = draw_class_on_image(("%s - %.2f" % (class_name, accuracy * 100) + "%"), image, (10, 80))
    return image    


cap = cv2.VideoCapture(0)
model, class_names = load_model(classdir)
with faceMesh:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # print(frame.shape)
        if not ret:
            continue
        Key = cv2.waitKey(1) & 0xFF
        prev_frame_time = time.time()
        frame = predict(frame, model)
        fps = 1 / (time.time() - prev_frame_time)
        cv2.putText(frame, "Fps: %d" % (fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.imshow("Recognition", frame)
        if Key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
