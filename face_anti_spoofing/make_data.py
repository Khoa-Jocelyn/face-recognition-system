import cv2

cap = cv2.VideoCapture(-1)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
label = "Face-Turn-Right"
outputdir = "videos"     
OK = 0
counter = 0
count_frame = 0
num_of_timesteps = 1
video = cv2.VideoWriter(outputdir + '/' + label + '.avi',
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                        (FRAME_WIDTH, FRAME_HEIGHT))
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))          
    x, y, z = frame.shape
    a = round(y / 4)
    b = round(x / 5)
    Key = cv2.waitKey(1)
    if  Key == ord(' '):
        OK += 1
    if OK == 1:
        video.write(frame)
        counter += 1
        count_frame += 1
    if counter >= num_of_timesteps:
        counter = 0
        OK = 0
    if count_frame >= 1500:
        break                
    cv2.rectangle(frame, (a, b), (3 * a, 4 * b), (0, 255, 0), 2)
    cv2.putText(frame, str(counter), (10, 40), 2, 1, (0, 255, 0), 2);
    cv2.putText(frame, str(count_frame), (10, 80), 2, 1, (0, 0, 255), 2);
    cv2.imshow(label, frame)
    if Key == ord('q'):
        break
