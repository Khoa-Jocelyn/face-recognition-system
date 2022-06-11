import cv2
import time
import mediapipe as mp
import os
from FaceMesh import FaceMeshDetector


def fancyDraw(img, bbox, l=20, t=3, rt=1):
    x, y, w, h = bbox
    x_max, y_max = x + w, y + h

    cv2.rectangle(img, bbox, (0, 255, 0), rt)

    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
    cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)

    # Top Right  x_max,y
    cv2.line(img, (x_max, y), (x_max - l, y), (0, 255, 0), t)
    cv2.line(img, (x_max, y), (x_max, y + l), (0, 255, 0), t)

    # Bottom Left  x,y_max
    cv2.line(img, (x, y_max), (x + l, y_max), (0, 255, 0), t)
    cv2.line(img, (x, y_max), (x, y_max - l), (0, 255, 0), t)

    # Bottom Right  x_max,y_max
    cv2.line(img, (x_max, y_max), (x_max - l, y_max), (0, 255, 0), t)
    cv2.line(img, (x_max, y_max), (x_max, y_max - l), (0, 255, 0), t)
    return img


class FaceDetector(object):
    def __init__(self, staticImageMode=True, maxNumFaces=1, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.img = None
        self.staticImageMode = staticImageMode
        self.maxNumFaces = maxNumFaces
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.faceMesh = FaceMeshDetector(
            self.staticImageMode,
            self.maxNumFaces,
            self.minDetectionConfidence,
            self.minTrackingConfidence
        )

    def findFaces(self, img, face_draw=True, mesh_draw=True):
        bounding_boxes = []
        img, faces = self.faceMesh.findFaceMesh(img, draw=mesh_draw)
        if len(faces) != 0:
            for face in faces:
                x_min = (face[0])[0]
                y_min = (face[0])[1]
                x_max = (face[0])[0]
                y_max = (face[0])[1]
                for lm in face:
                    if lm[0] < x_min:
                        x_min = lm[0]
                    if lm[1] < y_min:
                        y_min = lm[1]
                    if lm[0] > x_max:
                        x_max = lm[0]
                    if lm[1] > y_max:
                        y_max = lm[1]
                bounding_box = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                # print(bounding_box)
                bounding_boxes.append(bounding_box)
                if face_draw:
                    img = fancyDraw(img, bounding_box)
        return img, bounding_boxes, faces


def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    preTime = 0
    detector = FaceDetector(maxNumFaces=2)
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            img, bounding_boxes, faces = detector.findFaces(img, face_draw=True, mesh_draw=True)

            if cv2.waitKey(1) % 256 == 27:
                # ESC pressed
                break

            curTime = time.time()
            fps = 1 / (curTime - preTime)
            preTime = curTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Face detection", img)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
