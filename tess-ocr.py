import requests
from io import BytesIO
from PIL import Image
import logging
import os
import boto3
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
import pytesseract as pt
# OCR configuration
OCR_config = "--oem 3 --psm 6"

# Environment variables for accessing Label Studio and AWS S3
LABEL_STUDIO_ACCESS_TOKEN = "247460ee84c396560827ce0b2362873d6cd30594"
LABEL_STUDIO_HOST         = "http://65.0.197.237:8080"


class BBOXOCR(LabelStudioMLBase):

    @staticmethod
    def load_image(img_path_url):
        # Load an image from S3 or via HTTP(S), handling authorization for Label Studio server

        if img_path_url.startswith(('http://', 'https://')):
            headers = {"Authorization": "Token " + LABEL_STUDIO_ACCESS_TOKEN}
            response = requests.get(img_path_url, headers=headers)
            response.raise_for_status()
            image_content = response.content
            return Image.open(BytesIO(image_content))
        else:
            # Assume img_path_url is a local path within Label Studio's environment
            # You might need to adapt this part based on your setup
            filepath = get_image_local_path(img_path_url,
                                            label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                                            label_studio_host=LABEL_STUDIO_HOST)
            return Image.open(filepath)

    def predict(self, tasks, **kwargs):
        if not tasks:
            return []

        task = tasks[0]
        img_path_url = task["data"]["ocr"]

        if img_path_url.startswith(('http://', 'https://')):
            # Directly load from URL
            headers = {"Authorization": "Token " + LABEL_STUDIO_ACCESS_TOKEN}
            response = requests.get(img_path_url, headers=headers)
        else:
            # Construct full URL for non-http(s) cases
            file_url = LABEL_STUDIO_HOST + img_path_url
            headers = {"Authorization": "Token " + LABEL_STUDIO_ACCESS_TOKEN}
            response = requests.get(file_url, headers=headers)

        response.raise_for_status()
        image_content = response.content
        IMG = Image.open(BytesIO(image_content))
        text = pt.image_to_string(IMG,config=OCR_config)
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove single-letter words that are not 'a' or 'I'
        cleaned_text = re.sub(r'\b[^aI\s][\s$]', '', cleaned_text)

        result = {
            "result": [{
                "from_name": "answer",  # This should match the 'name' attribute of the <TextArea> tag.
                "to_name": "image",  # This should match the 'name' attribute of the <Image> tag.
                "type": "textarea",  # This indicates the type of annotation to be returned.
                "value": {
                    "text": [cleaned_text],  # The OCR extracted text.
                },
            }],
            "score": 1  # The confidence score, which can be adjusted based on OCR confidence.
        }
        return [result]
"""