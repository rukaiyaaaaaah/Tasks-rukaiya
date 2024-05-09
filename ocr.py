from PIL import Image
import io
import pytesseract as pt
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
import logging
import os
import json
import boto3

logger = logging.getLogger(__name__)
global OCR_config
OCR_config = "--psm 6"

LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST         = os.environ.get("LABEL_STUDIO_HOST")

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN     = os.environ.get("AWS_SESSION_TOKEN")
AWS_ENDPOINT          = os.environ.get("AWS_ENDPOINT")

S3_TARGET = boto3.resource('s3',
        endpoint_url=AWS_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        config=boto3.session.Config(signature_version='s3v4'),
        verify=False)

class BBOXOCR(LabelStudioMLBase):

    @staticmethod
    def load_image(img_path_url):
        # load an s3 image, this is very basic demonstration code
        # you may need to modify to fit your own needs
        if img_path_url.startswith("s3:"):
            bucket_name = img_path_url.split("/")[2]
            key = "/".join(img_path_url.split("/")[3:])

            obj = S3_TARGET.Object(bucket_name, key).get()
            data =  obj['Body'].read()
            image = Image.open(io.BytesIO(data))
            return image
        else:
            filepath = get_image_local_path(img_path_url,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST)
            return  Image.open(filepath)


    def predict(self, tasks, **kwargs):
        task = tasks[0]
        img_path_url = task["data"][value]

        IMG = self.load_image(img_path_url)

        result_text = pt.image_to_string(IMG, config=OCR_config).strip()
        result = {
            "result": [{
                "value": {
                    "text": [result_text],
                    "x": 0,  # Indicates the full image.
                    "y": 0,  # Indicates the full image.
                    "width": 100,  # Indicates the full image width as a percentage.
                    "height": 100,  # Indicates the full image height as a percentage.
                },
                "from_name": "text",  # This should match your label config in Label Studio.
                "to_name": "image",
                "type": "textarea",
            }],
            "score": 1  # Score can be adjusted based on OCR confidence if available.
        }
        return [result]

