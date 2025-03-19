import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer

def expand_polygon(bbox, delta):
    """
    Expands a polygon outward by a fixed distance while maintaining its shape.
    
    bbox: List of (x, y) points forming the polygon
    delta: Distance to expand outward
    
    Returns: Expanded bounding box as a NumPy array
    """
    box_np = np.array(bbox, np.float32).reshape(-1, 2)  # Convert to NumPy array
    centroid = np.mean(box_np, axis=0)  # Compute the centroid (center)
    
    # Compute vectors from centroid to each point
    vectors = box_np - centroid

    # Normalize vectors to maintain direction
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Expand outward
    expanded_box = box_np + unit_vectors * delta

    a = math.sqrt((box_np[0][0]- box_np[1][0])**2 + (box_np[0][1]- box_np[1][1])**2)
    a_exp = math.sqrt((expanded_box[0][0]- expanded_box[1][0])**2 + (expanded_box[0][1]- expanded_box[1][1])**2)
    b = math.sqrt((box_np[1][0]- box_np[2][0])**2 + (box_np[1][1]- box_np[2][1])**2)
    b_exp = math.sqrt((expanded_box[1][0]- expanded_box[2][0])**2 + (expanded_box[1][1]- expanded_box[2][1])**2)
    print(a)
    print(a_exp)
    print(b)
    print(b_exp)
    print(box_np)
    print(expanded_box)

    return expanded_box.astype(np.int32)

def scale_polygon(bbox, scale_factor):
    """
    Scales a polygon outward from its centroid while maintaining its shape.

    bbox: List of (x, y) points forming the polygon
    scale_factor: How much to increase the size (e.g., 2.0 means double)

    Returns: Scaled bounding box as a NumPy array
    """
    box_np = np.array(bbox, np.float32).reshape(-1, 2)  # Convert to NumPy array
    centroid = np.mean(box_np, axis=0)  # Compute the centroid (center)
    
    # Compute vectors from centroid to each point
    vectors = box_np - centroid

    # Scale vectors outward
    expanded_box = centroid + vectors * scale_factor  # Multiply by scale factor

    return expanded_box.astype(np.int32)

def merge_bounding_boxes(bboxes, threshold=20):
    """
    Recursively merges bounding boxes that are close together.

    bboxes: List of bounding boxes (each as a list of four [x, y] points)
    threshold: Maximum distance between boxes to consider them as one group

    Returns: List of merged bounding boxes.
    """
    if not bboxes:
        return []

    # Convert bboxes to 2D NumPy arrays
    bboxes = [np.array(bbox, dtype=np.float32) for bbox in bboxes]

    merged = []
    used = set()

    def get_connected_boxes(idx, group):
        """ Recursively find all connected bounding boxes """
        for j, bbox in enumerate(bboxes):
            if j in used:
                continue

            # Compute pairwise distance
            min_distance = np.min(np.linalg.norm(group[:, None, :] - bbox[None, :, :], axis=-1))

            if min_distance < threshold:  # If distance is less than threshold, merge
                used.add(j)
                group = np.vstack((group, bbox))  # Add to group
                group = get_connected_boxes(j, group)  # Recursive call to merge further
        return group

    for i, bbox in enumerate(bboxes):
        if i in used:
            continue
        
        used.add(i)
        merged_box = get_connected_boxes(i, bbox)
        
        # Compute the enclosing bounding box (axis-aligned)
        # Find the min/max coordinates from the merged group of boxes
        merged_box = np.array(merged_box)  # Ensure merged_box is an array
        x_min, y_min = np.min(merged_box, axis=0)  # Get the smallest x, y
        x_max, y_max = np.max(merged_box, axis=0)  # Get the largest x, y

        # Add the merged bounding box to the result
        merged.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    return merged



# Initialize detector
detector = MMOCRInferencer(det='dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015')

# Perform inference
img_path = "/home/fgervino-iit.local/text_dec_and_rec/resized11.jpg"
result = detector(img_path)

# Load image
image = cv2.imread(img_path)

# Extract bounding boxes
predictions = result['predictions'][0]  # List of detected bounding boxes
bboxes = predictions['det_polygons']

delta = 50
scale_factor = 1

threshold = 20
bboxes = merge_bounding_boxes(bboxes, threshold)

# Crop the image using bounding boxes
for bbox in bboxes:

    # Create a blank mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    #expanded_box = expand_polygon(bbox, delta)
    expanded_box = scale_polygon(bbox, scale_factor)

    # Fill the mask with the region inside the bbox
    cv2.fillPoly(mask, [expanded_box], 255)

    # Extract the region
    cropped_region = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow(f"Cropped Region", cropped_region)
    cv2.waitKey(0)  # Waits for key press before moving to next cropped image
    cv2.destroyAllWindows()