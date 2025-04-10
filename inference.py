import tritonclient.http as httpclient
import numpy as np
import cv2
import threading
import time

# Model configuration
url = "localhost:8000"  # No scheme like "http://"
model_name = "yolov7tiny"
model_version = "1"

# Load the input image and store original dimensions
image_path = 'input_image.jpg'
image = cv2.imread(image_path)
original_image = image.copy()  # Keep a copy to draw detections later
orig_height, orig_width = original_image.shape[:2]

# Resize and preprocess image for the model
image = cv2.resize(image, (640, 640))  # Resize to model input size
image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
image = np.transpose(image, (2, 0, 1))  # Convert from (H, W, C) to (C, H, W)
image = np.expand_dims(image, axis=0)  # Add batch dimension [1, C, H, W]

# Connect to Triton Inference Server
client = httpclient.InferenceServerClient(url=url)

# Set up input tensor
inputs = [
    httpclient.InferInput("images", image.shape, "FP32")
]
inputs[0].set_data_from_numpy(image)

# Set up output tensors to be fetched
outputs = [
    httpclient.InferRequestedOutput("num_dets"),
    httpclient.InferRequestedOutput("det_boxes"),
    httpclient.InferRequestedOutput("det_scores"),
    httpclient.InferRequestedOutput("det_classes")
]

# Confidence threshold for detections
threshold = 0.60

# IoU threshold for NMS
iou_threshold = 0.5

# Non-Maximum Suppression (NMS) function
def non_max_suppression(boxes, scores, iou_threshold):
    boxes = np.array(boxes)
    scores = np.array(scores)
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - x
    h = boxes[:, 3] - y
    boxes = np.vstack((x, y, w, h)).transpose()
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)
    return indices

# Function to process model output and apply thresholding
def process_detections(response, thread_boxes, thread_scores):
    num_dets = response.as_numpy("num_dets")[0].item()
    det_boxes = response.as_numpy("det_boxes")
    det_scores = response.as_numpy("det_scores")
    det_classes = response.as_numpy("det_classes")

    for i in range(int(num_dets)):
        score = det_scores[0][i]
        if score >= threshold:
            box = det_boxes[0][i]  # [x_min, y_min, x_max, y_max]
            thread_boxes.append(box.tolist())
            thread_scores.append(float(score))

# Inference function for each thread
def infer_thread(thread_id, interval, thread_boxes, thread_scores):
    print(f"Thread {thread_id} started.")
    while True:
        response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs, model_version=model_version)
        process_detections(response, thread_boxes, thread_scores)
        print(f"Thread {thread_id} completed an inference.")
        time.sleep(interval)

def main(num_threads, interval):
    threads = []
    all_boxes = []
    all_scores = []
    thread_boxes_list = [[] for _ in range(num_threads)]
    thread_scores_list = [[] for _ in range(num_threads)]

    for i in range(num_threads):
        t = threading.Thread(target=infer_thread, args=(i, interval, thread_boxes_list[i], thread_scores_list[i]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Combine results from all threads
    for i in range(num_threads):
        all_boxes.extend(thread_boxes_list[i])
        all_scores.extend(thread_scores_list[i])

    # Apply NMS
    indices = non_max_suppression(all_boxes, all_scores, iou_threshold)

    # Draw detections on the original image after NMS
    for i in indices.flatten():
        box = all_boxes[i]
        score = all_scores[i]
        class_id = int(det_classes[0][i])  # Use class id from any response

        # Scale back to original image size
        x_min, y_min, x_max, y_max = box
        x_min = int(x_min * orig_width / 640)
        y_min = int(y_min * orig_height / 640)
        x_max = int(x_max * orig_width / 640)
        y_max = int(y_max * orig_height / 640)

        # Draw bounding box and label
        cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(original_image, f"Class {class_id} {score:.2f}", (x_min, y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save the result image (optional)
    cv2.imwrite('detection_result_nms.jpg', original_image)

if __name__ == "__main__":
    num_threads = int(input("Enter the number of threads: "))
    interval = float(input("Enter the interval between requests (in seconds): "))
    main(num_threads, interval)
