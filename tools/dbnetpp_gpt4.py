from openai import AzureOpenAI
import base64
import cv2

import numpy as np
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer

from io import BytesIO
from PIL import Image

AZURE_API_KEY="94ec2d76955a48488952c81f0d591e94"
AZURE_ENDPOINT="https://iitlines-swecentral1.openai.azure.com/"

class TextInImage() :
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=f"{AZURE_ENDPOINT}", #do not add "/openai" at the end here because this will be automatically added by this SDK
            api_key=AZURE_API_KEY,
            api_version="2024-10-21"
        )

    def encode_image_array(self, image_array):
        """Converts a NumPy image (OpenCV format) to base64."""
        # Convert BGR (OpenCV format) to RGB (PIL format)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Save as JPEG in memory
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode in Base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def det_and_rec(self, image_array):
        """Sends image to GPT-4 Vision and retrieves detected text with bounding boxes."""
        try:
            base64_image = self.encode_image_array(image_array)

            messages = [
                {"role": "system", "content": 
                """
                Extract text from the provided image.
                The output will be a python list with the recognised texts.
                For example: ["text 1", "text 2", "text 3"].
                """},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is an image. Please extract any visible text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]       

            response = self.client.chat.completions.create(
                model="hsp-Vocalinteraction_gpt4o",
                messages=messages
            )

            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error processing image: {str(e)}"
        
    
def crop_image(img_path, detector,delta) :

    # Perform inference
    result = detector(img_path)

    # Load image
    image = cv2.imread(img_path)

    # Extract bounding boxes
    predictions = result['predictions'][0]  # List of detected bounding boxes
    bboxes = predictions['det_polygons']

    # Create a blank mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Crop the image using bounding boxes
    for bbox in bboxes:

        # Convert to NumPy array and reshape into coordinates
        box_np = np.array(bbox, np.int32).reshape(-1, 2)

        # Compute the center of the box
        center = np.mean(box_np, axis=0)

        # Expand each point outward
        expanded_box = box_np + np.sign(box_np - center) * delta

        # Reshape back to OpenCV format if needed
        expanded_box = expanded_box.astype(np.int32).reshape(-1, 1, 2)

        # Fill the mask with the region inside the bbox
        cv2.fillPoly(mask, [expanded_box], 255)

        # Extract the region
        cropped = cv2.bitwise_and(image, image, mask=mask)
    
    return cropped

bro = TextInImage()

# Initialize detector
detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015')
delta = 10

for k in range(1,15) :
    image_path = "/home/fgervino-iit.local/text_dec_and_rec/resized{}.jpg".format(k)

    cropped_image = crop_image(image_path,detector,delta)

    texts = bro.det_and_rec(cropped_image)

    print(texts)