import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer

# Initialize detector
detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015')

# Perform inference
img_path = "/home/fgervino-iit.local/text_dec_and_rec/resized10.jpg"
result = detector(img_path)

# Load image
image = cv2.imread(img_path)

# Extract bounding boxes
predictions = result['predictions'][0]  # List of detected bounding boxes
bboxes = predictions['det_polygons']

delta = 10
# Draw bounding boxes on the image
for bbox in bboxes:

    '''
    # Expand bbox
    bbox[0] = max(0,bbox[0]-delta); bbox[1] = max(0,bbox[1]-delta)
    bbox[2] = min(639,bbox[2]+delta); bbox[3] = max(0,bbox[3]-delta)
    bbox[4] = min(639,bbox[4]+delta); bbox[5] = min(479,bbox[5]+delta)
    bbox[6] = max(0,bbox[6]-delta); bbox[7] = min(479,bbox[7]+delta)
    '''

    # Convert to NumPy array and reshape into coordinates
    box_np = np.array(bbox, np.int32).reshape(-1, 2)

    # Compute the center of the box
    center = np.mean(box_np, axis=0)

    # Expand each point outward
    expanded_box = box_np + np.sign(box_np - center) * delta

    # Reshape back to OpenCV format if needed
    expanded_box = expanded_box.astype(np.int32).reshape(-1, 1, 2)

    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [expanded_box], isClosed=True, color=(0, 255, 0), thickness=2)

# Save the image with bounding boxes
#output_path = "output_with_boxes.jpg"
#cv2.imwrite(output_path, image)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()