# Mastering AI Request Volumes: Scalable Solutions for High and Low Demands

In this guide, you'll learn how to build a scalable AI inference system that dynamically handles fluctuating workloads. Using tools like Docker, Kubernetes, and Triton Inference Server, this step-by-step tutorial covers everything from installation to horizontal scaling.

---
![](images/triton-server-hps_architecture.png?raw=true)
*Figure: Example output of the detection model after inference.*
---
# Table of Contents

- [Mastering AI Request Volumes: Scalable Solutions for High and Low Demands](#mastering-ai-request-volumes-scalable-solutions-for-high-and-low-demands)
  - [1. Create simple Vision based AI Model Application](#1-create-simple-vision-based-ai-model-application)
    - [1.1 Installation](#11-installation)
      - [1.1.1 NVIDIA Container Toolkit](#111-nvidia-container-toolkit)
      - [1.1.2 K8s - Minikube](#112-k8s---minikube)
      - [1.1.3 Kubectl](#113-kubectl)
      - [1.1.4 Helm and GPU Operator](#114-helm-and-gpu-operator)
    - [1.2 Preparing the YOLOv7 AI Model](#12-preparing-the-yolov7-ai-model)
    - [1.3 Deploying Triton Inference Server](#13-deploying-triton-inference-server)
      - [Deployment Configuration](#deployment-configuration)
      - [Triton Service Configuration](#triton-service-configuration)
      - [Verify Model Deployment](#verify-model-deployment)
    - [1.4 Create High GPU Usage and Check Results](#14-create-high-gpu-usage-and-check-results)
  - [2. Manage Demands with Horizontal Pod Autoscale](#2-manage-demands-with-horizontal-pod-autoscale)
    - [2.1 Install DCGM on Host](#21-install-dcgm-on-host)
    - [2.2 Deploy DCGM Exporter](#22-deploy-dcgm-exporter)
    - [2.3 Set Up Prometheus and Prometheus Adapter](#23-set-up-prometheus-and-prometheus-adapter)
    - [2.4 Configure Horizontal Pod Autoscaler (HPA)](#24-configure-horizontal-pod-autoscaler-hpa)
- [Acknowledgements](#acknowledgements)
- [References](#references)
    
---

## 1. Create simple Vision based AI Model Application

### 1.1 Installation

#### 1.1.1 NVIDIA Container Toolkit

Install the NVIDIA Container Toolkit to enable GPU support for Docker containers.

```bash
# Add NVIDIA's GPG key to the system's keyring
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the NVIDIA Container Toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Enable experimental features and install the toolkit
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

#### Verify Installation:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu22.04 nvidia-smi
```
Expected Output: The nvidia-smi command should display GPU details.



---
### 1.1.2 K8s - Minikube

Install and configure Minikube with GPU support.

```bash
# Download and install Minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

# Start Minikube with GPU and volume mounting enabled
minikube start --driver docker --container-runtime docker --gpus all --force --mount --mount-string="/mnt/triton_models:/mnt/triton_models"
```

#### Verify Installation:
```bash
minikube status
```

---

### 1.1.3 Kubectl

Install `kubectl`, the Kubernetes command-line tool.

```bash
# Download and install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

#### Verify Installation:
```bash
kubectl get pods -A
```

---

### 1.1.4 Helm and GPU Operator

#### Install Helm:
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

#### Install GPU Operator:
```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace default \
  --set operator.defaultRuntime=docker \
  --set toolkit.enabled=true \
  --set devicePlugin.config.name=time-slicing-config
```

#### Verify Setup with a Test Pod:
Create a test pod to validate GPU Operator functionality.

`cuda-test-pod.yaml`
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

Deploy and verify:
```bash
kubectl apply -f cuda-test-pod.yaml
kubectl exec -it gpu-test -- bash
nvidia-smi
```

#### Time Slicing:
`time-slicing-config.yaml`
```
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: default
data:
  default: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 10
```

```bash
kubectl apply -f time-slicing-config.yaml
kubectl patch clusterpolicy cluster-policy \
  --type merge \
  -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config", "default": "default"}}}}'
```


> **Note:** Restart 'nvidia-device-plugin' pod and test multiple GPU deployment 

---

## 1.2 Preparing the YOLOv7 AI Model

1. **Download the YOLOv7-tiny model:**
   ```bash
   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
   ```

2. **Export the YOLOv7 model to ONNX format:**
   ```bash
   # download export.py -> https://github.com/WongKinYiu/yolov7
   python3 export.py --weights ./yolov7-tiny.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
   ```

3. **Optimize the ONNX model with TensorRT:**
   ```bash
   docker pull nvcr.io/nvidia/tensorrt:23.09-py3
   docker run --rm -it --gpus all nvcr.io/nvidia/tensorrt:23.09-py3

   # Replace <container_id> with actual container ID
   docker cp ./yolov7-tiny.onnx <container_id>:/home
   /usr/src/tensorrt/bin/trtexec --onnx=./yolov7-tiny.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --fp16 --workspace=4096 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache
   ```

4. **Move the optimized model to the Triton model repository:**
   ```bash
   mkdir -p /mnt/triton_models/yolov7tiny/1/
   mv yolov7-fp16-1x8x8.engine /mnt/triton_models/yolov7tiny/1/model.plan
   ```

---

## 1.3 Deploying Triton Inference Server

### Deployment Configuration

Create a deployment for Triton Inference Server.

`triton-deployment.yaml`
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
        ports:
          - containerPort: 8000
          - containerPort: 8001
          - containerPort: 8002
        resources:
          limits:
            nvidia.com/gpu: 1
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

`triton-service.yaml`
```yaml
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
    nodePort: 30001  # Dışarıya açık olacak port
  - name: grpc
    protocol: TCP
    port: 8001
    targetPort: 8001
    nodePort: 30002
  - name: metrics
    protocol: TCP
    port: 8002
    targetPort: 8002
    nodePort: 30003
  type: NodePort
```

### Port-forward Triton service to local port 8000
```bash
kubectl port-forward svc/triton-service 8000:8000 &
```

---

### Verify Model Deployment

Use `curl` to verify the deployed model:
```bash
curl -X GET http://localhost:8000/v2/models/yolov7tiny
```

#### Expected Output:
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

---

### 1.4 Create High GPU Usage and Check Results

#### Create GPU Usage:
```bash
python3 inference.py # example: thread 99, sleep 0.001
watch -n 1 nvidia-smi # run another session
```

#### Check Inference Result:
![](images/detection_result_nms.jpg?raw=true)
*Figure: Example output of the detection model after inference.*

---

## 2. Manage Demands with Horizontal Pod Autoscale

### 2.1 Install DCGM on Host

```bash
# Download the NVIDIA CUDA keyring package
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb

# Install the CUDA keyring package
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Add the CUDA repository
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Update package list and install DCGM
sudo apt-get update && sudo apt-get install -y datacenter-gpu-manager

# Start and enable NVIDIA DCGM service
sudo systemctl --now enable nvidia-dcgm
sudo systemctl status nvidia-dcgm

# Discover GPUs with DCGM
dcgmi discovery -l
```

### 2.2 Deploy DCGM Exporter

```bash
# Add GPU-Helm-Charts repository
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts

# Update the repository
helm repo update

# Install DCGM Exporter
helm install --generate-name gpu-helm-charts/dcgm-exporter

# Apply DCGM Exporter YAML file to Kubernetes
kubectl create -f https://raw.githubusercontent.com/NVIDIA/dcgm-exporter/master/dcgm-exporter.yaml

# Get the name of the first DCGM Exporter pod
NAME=$(kubectl get pods -l "app.kubernetes.io/name=dcgm-exporter" -o "jsonpath={ .items[0].metadata.name}")

# Port forward the pod's 9400 port to local 8080
kubectl port-forward $NAME 8080:9400 &

# Access metrics at localhost:8080
curl -sL http://127.0.0.1:8080/metrics

# Fetch DCGM_FI_DEV_GPU_UTIL metric
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/services/dcgm-exporter-1744216530/DCGM_FI_DEV_GPU_UTIL" | jq .
```

---

### 2.3 Set Up Prometheus and Prometheus Adapter

```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community \
   https://prometheus-community.github.io/helm-charts

# Search for kube-prometheus in the repository
helm search repo kube-prometheus

# Export default values for kube-prometheus-stack
helm inspect values prometheus-community/kube-prometheus-stack > /tmp/kube-prometheus-stack.values

# Edit `kube-prometheus-stack.values` file

> **Note:** I shared the complete file. Please check the differences carefully.

# Install Prometheus stack with custom values
helm install prometheus-community/kube-prometheus-stack \
   --create-namespace --namespace prometheus \
   --generate-name \
   --values /tmp/kube-prometheus-stack.values

# Install Prometheus Adapter with required settings
helm install prometheus-adapter prometheus-community/prometheus-adapter \
   --namespace prometheus \
   --set rbac.create=true \
   --set prometheus.url=http://kube-prometheus-stack-1744-prometheus.prometheus.svc \
   --set prometheus.port=9090

```bash
# Port forward Prometheus service to local port 9090
kubectl port-forward svc/kube-prometheus-stack-1744-prometheus -n prometheus 9090:9090 &
```

```
# Check for the DCGM_FI_DEV_MEM_COPY_UTIL metric
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq -r . | grep DCGM_FI_DEV_MEM_COPY_UTIL
```

---

### 2.4 Configure Horizontal Pod Autoscaler (HPA)

Create an HPA YAML file to dynamically scale Triton Inference Server pods based on GPU utilization.

`hpa_gpu.yaml`
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-inference-server
  minReplicas: 1
  maxReplicas: 2
  metrics:
  - type: Object
    object:
      metric:
        name: DCGM_FI_DEV_GPU_UTIL
      describedObject:
        kind: Service
        name: dcgm-exporter-1744216530 # Servis adı
      target:
        type: Value
        value: '2'
```

Apply the HPA configuration:
```bash
kubectl apply -f hpa_gpu.yaml
```

Check the HPA status adn metrics:
```bash
kubectl get hpa triton-hpa
```

#### Check HPA Result:
![](images/result.png?raw=true)
*Figure: Example output of the HPA.*

---

## Acknowledgements 
I would like to thank my teammates for their valuable support during this work.

- **Ahmet Selim Demirel**
- **Doğan Mehmet Başoğlu**
- **Elif Cansu Ada**
- **Mevlüt Ardıç**
- **Serhat Karaca**

## References
- [NVIDIA NLP Scaling Documentation](https://docs.nvidia.com/ai-enterprise/deployment/natural-language-processing/latest/scaling.html)
- [Kubernetes Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Vision Model - Yolov7](https://github.com/WongKinYiu/yolov7)
