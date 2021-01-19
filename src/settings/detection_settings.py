MODEL_ROOT = '/tmp/shared/relabs/loyalty_coffee/'
FACEDETECT_XML = f'{MODEL_ROOT}recogniser/models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
MASKDETECTION_XML = f"{MODEL_ROOT}recogniser/models/mask/face_mask.xml"
AGE_GANDERDETECTION_XML = f"{MODEL_ROOT}recogniser/models/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
HEADPOSE_XML = f"{MODEL_ROOT}recogniser/models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
REIDENTIFICATION_XML = F"{MODEL_ROOT}recogniser/models/reidentification/face-reidentification-retail-0095.xml"
FPS = 1

#VIDEO_INPUT = 'rtsp://192.168.250.46:554/11'
VIDEO_INPUT = 'rtsp://192.168.250.186/ch0_0.h264'
#VIDEO_INPUT = 0
