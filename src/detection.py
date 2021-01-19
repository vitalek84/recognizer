
import platform
import time
import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore
from settings import detection_settings as settings
from settings import api_settings
from Models import MaskDetector, PoseEstimator, AgeGenderRecognizer, FaceDetector, Face2Vector
from api import api


def main():

    ie = IECore()
    face_detector = FaceDetector(ie, settings.FACEDETECT_XML)
    mask_detector = MaskDetector(ie, settings.MASKDETECTION_XML)
    age_gender_recognizer = AgeGenderRecognizer(
        ie, settings.AGE_GANDERDETECTION_XML)
    pose_estimator = PoseEstimator(ie, settings.HEADPOSE_XML)
    face2vector = Face2Vector(ie, settings.REIDENTIFICATION_XML)
    cap = cv.VideoCapture(settings.VIDEO_INPUT)
    prev = 0
    wait_key_time = 1
    while (cap.isOpened()):
        if cap:
            hasFrame, frame = cap.read()

        if not hasFrame:
            break

        time_elapsed = time.time() - prev

        if time_elapsed > (1./settings.FPS):
            prev = time.time()

            face_blob = face_detector.preprocessing(frame)
            face_detector.do_inference(face_blob)
            face_detector.postprocessing()
            faces = face_detector.get_faces(frame)

            if faces:
                # max face in frame

                face = max(faces, key=lambda face: face[1])[0]
                base64_face = api.face2base64(
                    cv.cvtColor(face, cv.COLOR_BGR2RGB))
                pose_blob = pose_estimator.preprocessing(face)
                pose_estimator.do_inference(pose_blob)

                if pose_estimator.is_frontal_face(delta=15):

                    mask_blob = mask_detector.preprocessing(face)
                    mask_detector.do_inference(
                        mask_blob)

                    if not mask_detector.is_mask_on():

                        age_gender_blob = age_gender_recognizer.preprocessing(
                            face)
                        age_gender_recognizer.do_inference(
                            age_gender_blob)
                        age, gender = age_gender_recognizer.postprocessing()

                        face_vector_blob = face2vector.preprocessing(face)
                        face2vector.do_inference(face_vector_blob)
                        face_vector = face2vector.postprocessing()

                        data = api.data2dict(api_settings.SHOP_ID,
                                             age, gender, ident=face_vector, face=base64_face)

                        api.send_data(data, api_settings.API_IDENT)

                    else:
                        data = api.data2dict(
                            api_settings.SHOP_ID, mask=True, face=base64_face)
                        api.send_data(data, api_settings.API_MASK)
                else:
                    data = api.data2dict(
                        api_settings.SHOP_ID, frontal_face=False, face=base64_face)
                    api.send_data(data, api_settings.API_LOOK2CAMER)

        key = cv.waitKey(wait_key_time)
        if key in {ord('q'), ord('Q')}:
            break


if __name__ == "__main__":
    main()
