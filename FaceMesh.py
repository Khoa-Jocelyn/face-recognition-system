import cv2
import time
import mediapipe as mp
import os


class FaceMeshDetector(object):
    def __init__(self, staticImageMode=True, maxNumFaces=1, minDetectionConfidence=0.5, minTrackingConfidence=0.5):

        self.results = None
        self.imgRGB = None
        self.staticImageMode = staticImageMode
        self.maxNumFaces = maxNumFaces
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh

        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticImageMode,
            max_num_faces=self.maxNumFaces,
            min_detection_confidence=self.minDetectionConfidence,
            min_tracking_confidence=self.minTrackingConfidence)

        self.drawSpec = self.mpDraw.DrawingSpec(
            thickness=1,
            circle_radius=1,
            color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:

            for faceLms in self.results.multi_face_landmarks:

                face = []
                # Xác đinh vị trí của từng landmark trong khuôn mặt
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), int(lm.z * ic)
                    # Show id face landmark
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec,
                        self.drawSpec)

        return img, faces
