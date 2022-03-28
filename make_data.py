import cv2
import time
import os
import detect_face
import tensorflow.compat.v1 as tf
from sys import exit
def main():
    lable = "DinhVanKhoa"
    npy = './npy'
    video_capture = cv2.VideoCapture(0)
    img_counter = 100
    copy = img_counter
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30
            threshold = [0.7, 0.8, 0.8]
            factor = 0.709
            tmp = 1
            while True:
                ret, frame = video_capture.read()
                h_frame, w_frame, _ = frame.shape
                frame = cv2.flip(frame, 1)
                copy_frame = frame
                y = round(h_frame/5)
                min_vector = (0, y*tmp)
                max_vector = (w_frame, y*(tmp+1))
                timer = time.time()
                bounding_boxes, key_points = detect_face.detect_face(
                    frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                key = cv2.waitKey(1)
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    for i in range(faceNum):
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        bbox = xmin, ymin, xmax, ymax
                        left_eye = (key_points[0][0], key_points[5][0])
                        right_eye = (key_points[1][0], key_points[6][0])
                        try:
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            if left_eye[1] >= (y*tmp) and right_eye[1] >= (y*tmp) and left_eye[1] <= (y*(tmp+1)) and right_eye[1] <= (y*(tmp+1)) :
                                if not os.path.exists('train_img/' + str(lable)):
                                    os.mkdir('train_img/' + str(lable))
                                else:
                                    if img_counter > 0:
                                        img_name = lable + "{}_{}.jpg".format(tmp, img_counter)
                                        save_path = "train_img/" + lable + "/" + img_name
                                        cv2.imwrite(save_path, frame)
                                        print("Output: {}".format(save_path))
                                        img_counter -= 1
                                    elif img_counter == 0:
                                        if tmp < 3:
                                            tmp += 1
                                            img_counter = copy
                                        elif tmp == 3:
                                            exit()
                        except Exception as e:
                            print("Error: ", e)
                cv2.rectangle(copy_frame, min_vector, max_vector, (0, 255, 0), 2)
                endtimer = time.time()
                fps = 1 / (endtimer - timer)
                cv2.putText(frame, "Fps: {:.2f}".format(
                    fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                cv2.imshow('Face Recognition', copy_frame)
                if key == 113:
                    break
            video_capture.release()
            cv2.destroyAllWindows()

        


if __name__ == "__main__":
    main()