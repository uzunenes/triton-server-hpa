## Mastering AI Request Volume: Scalable Solutions for High and Low Demand

This guide provides a comprehensive, step-by-step walkthrough for setting up a scalable AI inference environment. Leveraging Docker, Kubernetes (via Minikube), Triton Inference Server, and Python, you'll learn how to efficiently handle both high and low volumes of AI requests. Whether you're just setting up or aiming to scale, this guide will help ensure your infrastructure adapts seamlessly to varying workloads.

---

### 1 Installation and GPU Utilization

#### 1.1 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit
```

#### Check Installation:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu22.04 nvidia-smi
```

---

### 1.2 Minikube

```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

minikube start --driver docker --container-runtime docker --gpus all --force --mount --mount-string="/mnt/triton_models:/mnt/triton_models"
```

#### Check Installation:
```bash
minikube status
```

---

### 1.3 Kubectl

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

#### Check Installation:
```bash
kubectl get pods -A
```

---

### 1.4 Helm and GPU Operator

#### Install Helm:
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

#### Install GPU Operator:
```bash
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace default \
  --set operator.defaultRuntime=docker \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set mig.strategy=none
```

---

#### Check Installation: `cuda-test-pod.yaml`
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:12.2.0-devel-ubuntu22.04
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: Never
```

#### Verify GPU Operator works:
```bash
kubectl apply -f cuda-test-pod.yaml
kubectl exec -it gpu-test -- bash
nvidia-smi
```

---

### YOLOv7 AI Model Preparation

1. Download YOLOv7-tiny model:
   ```bash
   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
   ```
2. Export YOLOv7 model to ONNX:
   ```bash
   python3 export.py --weights ./yolov7-tiny.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
   ```

3. Optimize the ONNX model with TensorRT:
   ```bash
   docker pull nvcr.io/nvidia/tensorrt:23.09-py3
   docker run --rm -it --gpus all nvcr.io/nvidia/tensorrt:23.09-py3

   docker cp ./yolov7-tiny.onnx containerID:/home
   /usr/src/tensorrt/bin/trtexec --onnx=./yolov7-tiny.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --fp16 --workspace=4096 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache
   ```

4. Save the optimized model:
   ```bash
   download yolov7-fp16-1x8x8.engine host
   mkdir -p /mnt/tritonmodels/yolov7-tiny/1/
   mv yolov7-fp16-1x8x8.engine /mnt/tritonmodels/yolov7-tiny/1/model.plan
   ```

---

## Part 2: Deploy Triton Inference Server

### Deployment Configuration

#### `triton-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference-server
  labels:
    app: triton-inference-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: triton-inference-server
  template:
    metadata:
      labels:
        app: triton-inference-server
    spec:
      containers:
      - name: triton-inference-server
        image: nvcr.io/nvidia/tritonserver:23.09-py3
        args:
          - "tritonserver"
          - "--model-repository=/mnt/triton_models"
          - "--log-verbose=1"
          - "--strict-model-config=false"
        ports:
          - containerPort: 8000
          - containerPort: 8001
          - containerPort: 8002
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
          - name: NVIDIA_VISIBLE_DEVICES
            value: "all"
          - name: TRITONSERVER_PROM_METRICS
            value: "true"  # Enable Prometheus metrics
        volumeMounts:
          - name: model-repository
            mountPath: /mnt/triton_models
      volumes:
        - name: model-repository
          hostPath:
            path: /mnt/triton_models
            type: Directory
```

---

### Triton Service Configuration

#### `triton-service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: triton-service
  labels:
    app: triton-inference-server
spec:
  selector:
    app: triton-inference-server
  ports:
  - name: http
    protocol: TCP
    port: 8000
    targetPort: 8000
  - name: grpc
    protocol: TCP
    port: 8001
    targetPort: 8001
  - name: metrics
    protocol: TCP
    port: 8002
    targetPort: 8002
  type: LoadBalancer
```

---

### Check Model Deployment

1. Use `curl` to verify your model:
   ```bash
   curl -X GET http://localhost:8000/v2/models/yolov7tiny
   ```

#### Expected Result:
```json
{
  "name":"yolov7tiny",
  "versions":["1"],
  "platform":"tensorrt_plan",
  "inputs":[
    {"name":"images","datatype":"FP32","shape":[-1,3,640,640]}
  ],
  "outputs":[
    {"name":"num_dets","datatype":"INT32","shape":[-1,1]},
    {"name":"det_boxes","datatype":"FP32","shape":[-1,100,4]},
    {"name":"det_scores","datatype":"FP32","shape":[-1,100]},
    {"name":"det_classes","datatype":"INT32","shape":[-1,100]}
  ]
}
```
## Triton Inference Server Load Testing Script (Multithreaded)
``` python
import tritonclient.http as httpclient
import numpy as np
import cv2
import threading
import time

# Model configuration
url = "10.96.60.117:8000"  # No scheme like "http://"
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
```

#### check GPU usage with nvidia-smi
```bash
python3 inference.py
another bash-> watch -n 1 nvidia-smi
```
#### check image result
![](resul.jpg?raw=true)


# 2 Manage Demands with Horizontal Pod Autoscale
