
import sys
import logging
import os.path as osp
import cv2 as cv
import numpy as np


logging.basicConfig(format='[ %(levelname)s ] %(message)s',
                    level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


class Model:
    def __init__(self, ie, xml_file_path, bin_file_path=None,
                 device='CPU', plugin_config={}, max_num_requests=1,
                 results=None, caught_exceptions=None):
        self.ie = ie
        log.info('Reading network from IR...')
        if bin_file_path is None:
            bin_file_path = osp.splitext(xml_file_path)[0] + '.bin'
        self.net = ie.read_network(model=xml_file_path, weights=bin_file_path)

        log.info('Loading network to plugin...')
        self.exec_net = ie.load_network(
            network=self.net, device_name=device, config=plugin_config, num_requests=max_num_requests)

        self.input_key = self._get_input_key(self.net)
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_key].input_data.shape

    def get_input_shape(self):
        return self.n, self.c, self.h, self.w

    def _get_input_key(self, net):
        blob_name, _ = next(iter(net.input_info.items()))
        return blob_name

    def get_output_key(self, output):
        if len(output) > 1:
            return [blob_name for blob_name in output]
        blob_name, _ = next(iter(output.items()))
        return blob_name

    def preprocessing(self, image):
        try:
            n, c, h, w = self.get_input_shape()
            blob = cv.resize(image, (w, h))
            blob = blob.transpose((2, 0, 1))
            blob = blob.reshape((n, c, h, w))
            return blob
        except Exception as e:
            print("You are too close to camera")

    def do_inference(self, blob):
        res = self.exec_net.infer({self.input_key: blob})
        self.outputs = res

    def do_async_inference(self, blob, request_id=0):
        req_handle = self.exec_net.start_async(
            request_id=request_id, inputs={self.input_key: blob})
        status = req_handle.wait()
        blob = req_handle.output_blobs
        output_key = self.get_output_key(blob)
        self.outputs = blob[output_key].buffer


class PoseEstimator(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = None

    def is_frontal_face(self, delta):
        if self.outputs:
            if abs(self.outputs['angle_y_fc'][0]) > delta:
                return False
            #elif abs(self.outputs['angle_p_fc'][0]) > delta:
            #    return False
            elif abs(self.outputs['angle_r_fc'][0]) > delta:
                return False
            return True


class AgeGenderRecognizer(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = None
        self.gander = {1: "Male", 0: "Female"}

    def postprocessing(self):
        if self.outputs:
            age = round(self.outputs["age_conv3"].item() * 100, 2)
            gander = self.gander[np.argmax(self.outputs["prob"])]
        return age, gander


class MaskDetector(Model):
    trash_hold = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = None

    def is_mask_on(self):
        output_key = self.get_output_key(self.outputs)
        if self.outputs[output_key][0] >= self.trash_hold:
            return True
        return False


class FaceDetector(Model):
    trash_hold = 0.8

    class Detection:
        def __init__(self, xmin, ymin, xmax, ymax, score, class_id):
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax
            self.score = score
            self.class_id = class_id

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paddings = {"min": 50, "max": 30}
        self.outputs = []
        self.detections = []

    def postprocessing(self):
        output_key = self.get_output_key(self.outputs)
        self.detections = [self.Detection(xmin, ymin, xmax, ymax, score, label)
                           for _, label, score, xmin, ymin, xmax, ymax in self.outputs[output_key][0][0]]

    def get_faces(self, frame):
        faces = []
        if self.detections:
            for detection in self.detections:
                if detection.score > self.trash_hold:
                    xmin = int(detection.xmin *
                               frame.shape[1] - self.paddings["min"])
                    ymin = int(detection.ymin *
                               frame.shape[0] - self.paddings["min"])
                    xmax = int(detection.xmax *
                               frame.shape[1] + self.paddings["max"])
                    ymax = int(detection.ymax *
                               frame.shape[0] + self.paddings["max"])
                    face = frame[ymin:ymax, xmin:xmax]
                    area = face.shape[0] * face.shape[1]
                    faces.append((face, area))
            return faces


class Face2Vector(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = None

    def postprocessing(self):
        if self.outputs:
            outputs_key = self.get_output_key(self.outputs)
        return self.outputs[outputs_key].reshape(256).tolist()
