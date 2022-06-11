from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
from websocket import create_connection
import threading
import math
import mediapipe as mp
import random

video = 0
npy = './npy'
modeldir = './model/20180402-114759.pb'
face_classifier_filename = './class/classifier.pkl'
train_img = './aligned_img'
serverName = 'ahkiot.herokuapp.com'
link_list = []
node_size = 10
result_names = ''
server_message = ''
action_face_classifier_filename = 'face_anti_spoofing/class/'
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
                    # print('lm.x: ', lm.x)
                    c_lm.append(lm.y)
                    # print('lm.y: ', lm.y)
                    c_lm.append(lm.z)
                    # print('lm.z: ',lm.z)
            if draw_lms:
                mpDraw.draw_landmarks(image,
                          lms,
                          mpFaceMesh.FACEMESH_CONTOURS,
                          drawSpec,
                          drawSpec)
    return c_lm, id


def draw_class_on_image(label, image, *args):
    cv2.putText(image, label, *args, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def predict(image, model):
    result_class = None
    image, results = mediapipe_detection(image, faceMesh)
    if results.multi_face_landmarks:
        c_lm, _ = extract_landmarks(image, results)
        if len(c_lm) == 384:
            c_lm = np.array(c_lm).reshape(1, -1)
            result_class = model.predict(c_lm)
            accuracy = model.score(c_lm, result_class)
            # print(json.dumps({'Class': class_name[0], 'Accuracy': '%.2f' % (accuracy)}, indent=4), '\n')
            if result_class != ['Normal']:
                image = draw_class_on_image(('%s - %.2f' % (result_class, accuracy * 100) + '%'), image, (10, 80))
    return image, result_class


def response_sever(serverName, message):
    global confirm
    ''' Connect to server '''
    ws = create_connection('ws://' + str(serverName), timeout=8)
    if ws:
        ''' Send data to the server '''
        print('Sending...')
        ws.send(message)
        print('Sent')
    
        ''' Receive message from server '''
        print('Receiving...')
        try:
            while ws.recv().decode('utf-8'):
                print(ws.recv().decode('utf-8'))
                if ws.recv().decode('utf-8') == 'close successfully':
                    mode['Recognition'] = True
                    mode['Anti-Spoofing'] = False
                    ws.close()
                    break
        except:
            print('Warning: Server disconected!')
            mode['Recognition'] = True
            mode['Anti-Spoofing'] = False
    ws.close()
        

def distance_eyes(left_eye, right_eye):
    w = math.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)
    return w


def accuracy_statistics(Nodes):
    count = 0
    if len(Nodes) == node_size:
        for Node in Nodes:
            if Node[1] >= 0.85:
                count += 1
    return count


def fancy_draw(img, bbox, l=30, t=3, rt=1):
    xmin, ymin, xmax, ymax = bbox
    # Bounding box
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), rt)
    # Top Left  x,y
    cv2.line(img, (xmin, ymin), (xmin + l, ymin), (0, 255, 0), t)
    cv2.line(img, (xmin, ymin), (xmin, ymin + l), (0, 255, 0), t)
    # Top Right  x_max,y
    cv2.line(img, (xmax, ymin), (xmax - l, ymin), (0, 255, 0), t)
    cv2.line(img, (xmax, ymin), (xmax, ymin + l), (0, 255, 0), t)
    # Bottom Left  x,y_max
    cv2.line(img, (xmin, ymax), (xmin + l, ymax), (0, 255, 0), t)
    cv2.line(img, (xmin, ymax), (xmin, ymax - l), (0, 255, 0), t)
    # Bottom Right  x_max,y_max
    cv2.line(img, (xmax, ymax), (xmax - l, ymax), (0, 255, 0), t)
    cv2.line(img, (xmax, ymax), (xmax, ymax - l), (0, 255, 0), t)
    return img


def keypoint_draw(image, key_points):
    cv2.circle(image, (int(key_points[0][i]), int(key_points[5][i])), 2, (0, 0, 255), -1)
    cv2.circle(image, (int(key_points[1][i]), int(key_points[6][i])), 2, (0, 0, 255), -1)
    cv2.circle(image, (int(key_points[2][i]), int(key_points[7][i])), 2, (0, 0, 255), -1)
    cv2.circle(image, (int(key_points[3][i]), int(key_points[8][i])), 2, (0, 0, 255), -1)
    cv2.circle(image, (int(key_points[4][i]), int(key_points[9][i])), 2, (0, 0, 255), -1)
    return image


start_time = 0
end_time = 0
confirm_str = ""
message = ''
counter = 0
challenge_request_result = set()
face_id = set()
mode = {'Recognition': True, 'Anti-Spoofing': False}
classdir = 'face_anti_spoofing/class/'
action_classifier_model, responce_class_names = load_model(classdir)
responce_class_names.pop()
challenge_request = ''
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30
        threshold = [0.7, 0.7, 0.7]
        factor = 0.709
        margin = 44
        image_size = 160
        input_image_size = 160
        human_names = os.listdir(train_img)
        human_names.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        embedding_size = embeddings.get_shape()[1]
        face_classifier_filename_exp = os.path.expanduser(face_classifier_filename)
        with open(face_classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')
        video_capture = cv2.VideoCapture(video)
        print('Start Recognition')
        while True:
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            x, y, z = frame.shape
            a = round(y / 4)
            b = round(x / 5)
            cv2.rectangle(frame, (0, 0), (y, b), (0, 255, 0), -1)
            fps_start_time = time.time()
            if mode['Anti-Spoofing'] == True:
                if counter <= 3:
                    message = ''
                    frame, challenge_responce = predict(frame, action_classifier_model)
                    end_time = time.time()
                    time_counter = end_time - start_time
                    confirm_str = "FACE_RECOGNITION_CONFIRM-" + result_names
                    if time_counter < 2:
                        if challenge_request == challenge_responce and challenge_responce not in ['', 'Normal']:
                            challenge_request_result.add(True)
                            mode['Recognition'] = True
                            mode['Anti-Spoofing'] = False
                            counter += 1
                    if time_counter > 2:
                        challenge_request_result.add(False)
                        mode['Recognition'] = True
                        mode['Anti-Spoofing'] = False
                        counter += 1
                if counter == 3:
                    if len(face_id) == 1 and len(challenge_request_result) == 1 and (True in challenge_request_result):
                        Thr_1 = threading.Thread(target=response_sever, args=(serverName, confirm_str))
                        Thr_1.start()
                        mode['Recognition'] = False
                        mode['Anti-Spoofing'] = False
                        message = 'Real'
                    else:
                        message = 'Fake'
                        mode['Recognition'] = True
                        mode['Anti-Spoofing'] = False
                    face_id.clear()
                    challenge_request_result.clear()
                    counter = 0
                print(face_id, challenge_request_result, message, counter)
                frame = draw_class_on_image(('%s - %.2f' % (challenge_request, 3 - time_counter)), frame, (10, 55))
            if mode['Recognition'] == True:
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, key_points = detect_face.detect_face(
                    frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if 0 < faceNum <= 1:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        bbox = xmin, ymin, xmax, ymax
                        # keypoint_draw(frame, key_points)
                        try:
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            if xmin >= a and xmax <= a * 3 and ymin >= b and ymax <= b * 4:
                                server_message = ''
                                cropped.append(frame[ymin:ymax, xmin:xmax,:])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(np.array(Image.fromarray(
                                    cropped[i]).resize((image_size, image_size))))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(
                                    scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                                feed_dict = {
                                    images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0,:] = sess.run(
                                    embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(
                                    len(best_class_indices)), best_class_indices]
                                fancy_draw(frame, bbox)
                                left_eye = (key_points[0][0], key_points[5][0])
                                right_eye = (key_points[1][0], key_points[6][0])
                                w = distance_eyes(left_eye, right_eye)
                                W = 6.3
                                x = [72, 62, 50, 45, 40, 35]  # x is w
                                y = [30, 40, 50, 60, 70, 80]  # y is the distance from eyes to webcam
                                A, B, C = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
                                d = A * w ** 2 + B * w + C
                                if best_class_probabilities > 0.80:
                                    link_list.append([human_names[best_class_indices[0]], best_class_probabilities])
                                    link_list = link_list[-node_size:]
                                    if accuracy_statistics(link_list) >= (node_size * (80 / 100)):
                                        result_names = human_names[best_class_indices[0]]
                                        face_id.add(human_names[best_class_indices[0]])
                                        if 25 <= d <= 35: 
                                            mode['Anti-Spoofing'] = True
                                            mode['Recognition'] = False
                                            start_time = time.time()
                                            challenge_request = random.choice(responce_class_names)
                                            link_list = []
                                    else:
                                        result_names = 'Unknown'
                                else:
                                    result_names = 'Unknown'
                                cv2.rectangle(frame, (xmin, ymin - 30),
                                                  (xmax, ymin - 10), (0, 255, 0), -1)
                                cv2.putText(frame, result_names + ' %.2f cm' % (d) , (xmin, ymin - 12), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)
                            else:
                                fancy_draw(frame, bbox)
                        except Exception as e:
                            print('Error: ', e)
                else:
                    link_list.clear()
            fps_end_time = time.time()
            fps = 1 / (fps_end_time - fps_start_time)
            frame = draw_class_on_image('%s' % (message), frame, (560, 30))
            frame = draw_class_on_image('Fps: %d' % (fps), frame, (10, 30))
            cv2.rectangle(frame, (a, b), (3 * a, 4 * b), (0, 255, 0), 2)
            cv2.imshow('Face Recognition', frame)
            key = cv2.waitKey(1)
            if key == 113:
                break
        video_capture.release()
        cv2.destroyAllWindows()
