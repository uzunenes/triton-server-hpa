# Mastering AI Request Volume: Scalable Solutions for High and Low Demand

This guide provides step-by-step instructions for setting up an environment to manage high or low volumes of AI requests using Docker, Kubernetes (MiniKube), Triton Inference Server, and Python. Follow these steps to ensure your AI models can scale efficiently based on demand.

---

## Part 1: Setting Up the Environment

### 1.1 NVIDIA Container Toolkit

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

### CUDA Test Pod Configuration

#### `cuda-test-pod.yaml`:
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

#### Verify CUDA:
```bash
kubectl exec -it gpu-test -- bash
nvidia-smi
```

---

### YOLOv7 Model Preparation

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
   download yolov7-fp16-1x8x8.engine file to /mnt/tritonmodels/yolov7-tiny/1/model.plan
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
