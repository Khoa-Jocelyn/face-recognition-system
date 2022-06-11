from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
import tkinter
from tkinter import *
from tkinter.ttk import *
import PIL.ImageTk
import cv2
import os
from FaceDetect import FaceDetector
import threading
from classifier import training
from preprocess import preprocesses
from websocket import create_connection

window = Tk()
window.title('Thêm Người Dùng')
window.geometry('700x480')
tmp = 1
img_counter = 100
copy = img_counter
fps = 0
label = ""
step = ''
Thr_1 = None
Thr_2 = None
Thr_3 = None
datadir = './aligned_img'
input_datadir = './train_img'
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
photo = None
serverName = 'ahkiot.herokuapp.com'
# Load video từ webcam
cap = cv2.VideoCapture(0)
# Cài đặt cho kích thước của webcam về theo kích thước mong muốn
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
preTime = 0
detector = FaceDetector(maxNumFaces=2)


def window_after(after_time, method_str):
    window.after(after_time, method_str)


def compute_eye_vector(img, face):
    # Tìm toạ độ trung tâm của mắt trái
    left_x = (face[133][0] + face[246][0]) / 2
    left_y = (face[133][1] + face[246][1]) / 2
    # Tìm toạ độ trung tâm của mắt phải
    right_x = (face[362][0] + face[466][0]) / 2
    right_y = (face[362][1] + face[466][1]) / 2
    return (int(left_x), int(left_y)), (int(right_x), int(right_y))


def response_sever(serverName, message):
    ''' Connect to server '''
    ws = create_connection('ws://' + str(serverName), timeout=8)
    if ws:
        ''' Send data to the server '''
        print('Sending...')
        ws.send(message)
        print('Sent')
    ws.close()


def Training():
    global datadir, input_datadir, modeldir, classifier_filename, step, tmp
    """[===== Data Preprocesses =====]"""
    Notify.configure(text="""
- Data Processing""")
    obj = preprocesses(input_datadir, datadir)
    nrof_images_total, nrof_successfully_aligned = obj.collect_data
    Notify.configure(text='''
- Total number of images: %d
- Number of successfully aligned images: %d 
- Training in progress, please don't turn off the program''' % (nrof_images_total, nrof_successfully_aligned))
    """[===== Data Training =====]"""
    obj = training(input_datadir, modeldir, classifier_filename)
    Notify.configure(text='''
- Saved classifier model to file "%s"
- All Done!''' % obj.main_train())
    txtNewUserName.configure(text=" ")
    step = ""
    tmp = 1
    img_counter = 100


def updateFrame():
    global photo, videoBox, preTime, fps, tmp, img_counter, copy, fps, step, label, txtNewUserName, Thr_1, Thr_2, Thr_3
    label = txtNewUserName.get()
    Thr_1 = threading.Thread(target=window_after, args=(1, updateFrame))
    Thr_2 = threading.Thread(target=Training, args=())
    Thr_3 = threading.Thread(target=response_sever, args=(serverName, "NewUser-%s"%label))
    if cap.isOpened() and step != "Training":
        success, frame = cap.read()
        h_frame, w_frame, _ = frame.shape
        # frame = cv2.flip(frame, 1)
        copy_frame = frame
        y = round(h_frame / 5)
        min_vector = (0, y * tmp)
        max_vector = (w_frame, y * (tmp + 1))
        if success:
            img = cv2.flip(frame, 1)
            img, bounding_boxes, faces = detector.findFaces(img, face_draw=True, mesh_draw=False)
            if len(faces) > 0:
                for face in faces:
                    left_eye, right_eye = compute_eye_vector(img, face)
                    cv2.circle(img, left_eye, 3, (0, 255, 0), -1)
                    cv2.circle(img, right_eye, 3, (0, 255, 0), -1)
                    try:
                        if list(left_eye)[1] >= (y * tmp) and list(right_eye)[1] >= (y * tmp) and list(left_eye)[1] <= (y * (tmp + 1)) and list(right_eye)[1] <= (y * (tmp + 1)):
                            if not os.path.exists('train_img/' + str(label)):
                                os.mkdir('train_img/' + str(label))
                            else:
                                if img_counter > 0:
                                    img_name = label + "{}_{}.jpg".format(tmp, img_counter)
                                    save_path = "train_img/" + label + "/" + img_name
                                    cv2.imwrite(save_path, copy_frame)
                                    print("Output: {}".format(save_path))
                                    img_counter -= 1
                                elif img_counter == 0:
                                    if tmp < 3:
                                        tmp += 1
                                        img_counter = copy
                                    elif tmp == 3:
                                        step = "Training"
                    except Exception as e:
                        print("Error: ", e)
                    curTime = time.time()
                    fps = 1 / (curTime - preTime)
                    preTime = curTime
                    cv2.rectangle(img, min_vector, max_vector, (0, 255, 0), 2)
                    cv2.rectangle(img, min_vector, max_vector, (0, 255, 0), 2)
                    Notify.configure(text=str(f'Fps: {int(fps)}'))
                    # Resize frame input
                    frame = cv2.resize(img, dsize=None, fx=1, fy=1)
                    # Chuyển đổi hệ màu BGR -> RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Chuyển frame sang ảnh tkinter
                    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    # Hiển Thị nên Canvas
                    videoBox.create_image(0, 0, image=photo, anchor=NW)
            else:
                Notify.configure(text="Can't Detect Face")
        Thr_2._stop()
        Thr_1.start()
    else:
        photo = None
        Thr_1._stop()
        Thr_2.start()
        Thr_3.start()
        cap.release()
        cv2.destroyAllWindows()


Container = Canvas(window, width=800, height=480, bg="#C0C0C0")
Container.pack()

videoBox = Canvas(Container, width=640, height=480, bg="#C0C0C0")
videoBox.place(x=160,)

Notify = tkinter.Label(videoBox, text="Notification", fg="green", font=("Arial", 12), bg="#C0C0C0", justify="left")
Notify.place(x=10, y=10)

newName = tkinter.Label(Container, text="Face Label:", fg="black", font=("Arial", 10), bg="#C0C0C0")
newName.place(height=20, x=10, y=10)

txtNewUserName = Entry(Container)
txtNewUserName.place(width=140, height=30, x=10, y=40)

bntfaceDetect = Button(Container, text="Start", command=updateFrame)
bntfaceDetect.place(width=140, height=40, x=10, y=80)

bntfaceCapture = Button(Container, text="Retrain", command=Training)
bntfaceCapture.place(width=140, height=40, x=10, y=130)

# bntfaceUpdate = Button(Container, text="Face update")
# bntfaceUpdate.place(width=140, height=40, x=10, y=180)

# bntfaceRecognition = Button(Container, text="Face recognition")
# bntfaceRecognition.place(width=140, height=40, x=10, y=230)

window.mainloop()
