import requests
import logging
from settings import api_settings as settings
import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO


def send_data(data, url):
    requests.adapters.DEFAULT_RETRIES = 100
    try:
        res = requests.post(url, json=data, headers=settings.HEADERS)
        return res
    except Exception as e:
        logging.exception(e)


def data2dict(shop, age=None, gender=None, ident=None, face=None, mask=False, frontal_face=True):
    return {
        "data": {
            "age": age,
            "gender": gender,
            "ident": ident,
            "shop": shop,
            "face": face,
            "mask": mask,
            "frontal_face": frontal_face
        }
    }


def face2base64(face):
    p_face = Image.fromarray(face)
    buffered = BytesIO()
    p_face.save(buffered, format="JPEG")
    encoded_face = base64.b64encode(buffered.getvalue())
    return encoded_face.decode('utf-8')
