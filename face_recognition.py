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

video = 0
npy = './npy'
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
train_img = "./aligned_img"
serverName = "autodoor.herokuapp.com"
link_list = []
node_size = 10
result_names = ""


def responseSever(serverName, message):
    """ Connect to server """
    ws = create_connection("ws://" + str(serverName))

    """ Send data to the server """
    print("Sending...")
    ws.send(message)
    print("Sent")

    """ Receive message from server """
    # print("Receiving...")
    # result = ws.recv()
    # print("Received '%s'" % result)

    ws.close()


def Distance_eyes(left_eye, right_eye):
    w = math.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)
    return w


def AccuracyStatistics(Nodes):
    count = 0
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


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30
        threshold = [0.7, 0.8, 0.8]
        factor = 0.709
        margin = 44
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')
        video_capture = cv2.VideoCapture(video)
        print('Start Recognition')
        while True:
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            x, y, z = frame.shape
            a = round(y / 3)
            bbox_img2 = cv2.rectangle(frame, (a, 0), (2 * a, x), (0, 255, 0), 2)
            timer = time.time()
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            bounding_boxes, key_points = detect_face.detect_face(
                frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]
            
            if faceNum > 0:
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
                    try:
                        if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                            print('Face is very close!')
                            continue
                        if xmin >= a and xmax <= a * 2:
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
                            w = int(Distance_eyes(left_eye, right_eye))
                            W = 6.3
                            x = [72, 62, 50, 45, 40, 35]  # x is w
                            y = [30, 40, 50, 60, 70, 80]  # y is the distance from eyes to webcam
                            A, B, C = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
                            d = A * w ** 2 + B * w + C
                            if best_class_probabilities > 0.80:
                                print(len(link_list), '\n')
                                print('i: ', i)
                                link_list.append(
                                    [HumanNames[best_class_indices[0]], best_class_probabilities])
                                link_list = link_list[-node_size:]
                                print(link_list)
                                result_names = ""
                                if AccuracyStatistics(link_list) >= (node_size * (80 / 100)):
                                    result_names = HumanNames[best_class_indices[0]]
                                else:
                                    result_names = "Unknown"
                            else:
                                result_names = "Unknown"
                            cv2.rectangle(frame, (xmin, ymin - 30),
                                              (xmax, ymin - 10), (0, 255, 0), -1)
                            cv2.putText(frame, result_names + f'-{int(d)}cm', (xmin, ymin - 12), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 0), thickness=1, lineType=1)
                        else:
                            fancy_draw(frame, bbox)
                    except Exception as e:
                        print("Error: ", e)
            else:
                link_list.clear()
            endtimer = time.time()
            fps = 1 / (endtimer - timer)
            cv2.putText(frame, "Fps: {:.2f}".format(
                fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.imshow('Face Recognition', frame)
            key = cv2.waitKey(1)
            if key == 113:
                break
        video_capture.release()
        cv2.destroyAllWindows()
