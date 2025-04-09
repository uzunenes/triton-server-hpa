
# Mastering AI Request Volume: Scalable Solutions for High and Low Demand

This guide provides a detailed, step-by-step process for setting up a scalable AI inference environment. By leveraging Docker, Kubernetes (via Minikube), Triton Inference Server, and Python, this guide equips you to efficiently handle both high and low volumes of AI requests. Whether you're starting fresh or scaling up, this guide ensures your infrastructure adapts effortlessly to fluctuating workloads.

---

## 1. Installation and GPU Utilization

### 1.1 NVIDIA Container Toolkit

Install the NVIDIA Container Toolkit to enable GPU support for Docker containers.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

#### Verify Installation:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-devel-ubuntu22.04 nvidia-smi
```

---

### 1.2 Minikube

Install and configure Minikube with GPU support.

```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

minikube start --driver docker --container-runtime docker --gpus all --force --mount --mount-string="/mnt/triton_models:/mnt/triton_models"
```

#### Verify Installation:
```bash
minikube status
```

---

### 1.3 Kubectl

Install `kubectl`, the Kubernetes command-line tool.

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

#### Verify Installation:
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

---

## 2. Preparing the YOLOv7 AI Model

1. **Download the YOLOv7-tiny model:**
   ```bash
   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
   ```

2. **Export the YOLOv7 model to ONNX format:**
   ```bash
   python3 export.py --weights ./yolov7-tiny.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
   ```

3. **Optimize the ONNX model with TensorRT:**
   ```bash
   docker pull nvcr.io/nvidia/tensorrt:23.09-py3
   docker run --rm -it --gpus all nvcr.io/nvidia/tensorrt:23.09-py3

   docker cp ./yolov7-tiny.onnx containerID:/home
   /usr/src/tensorrt/bin/trtexec --onnx=./yolov7-tiny.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --fp16 --workspace=4096 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache
   ```

4. **Save the optimized model:**
   ```bash
   mkdir -p /mnt/tritonmodels/yolov7-tiny/1/
   mv yolov7-fp16-1x8x8.engine /mnt/tritonmodels/yolov7-tiny/1/model.plan
   ```

---

## 3. Deploying Triton Inference Server

### Deployment Configuration

Create a deployment for Triton Inference Server.

`triton-deployment.yaml`
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
root@ubuntu:~# cat triton_dep.yaml
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

### Check GPU Usage and Results

#### Check GPU Usage:
```bash
python3 inference.py
another bash-> watch -n 1 nvidia-smi
```

#### Check Image Result:
![](resul.jpg?raw=true)

---

## 4. Manage Demands with Horizontal Pod Autoscale

### 4.1 Install DCGM and GPU Monitoring

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

# Discover GPUs with DCGM
dcgmi discovery -l
```

### 4.2 Deploy DCGM Exporter

```bash
# Add GPU-Helm-Charts repository
helm repo add gpu-helm-charts \
  https://nvidia.github.io/dcgm-exporter/helm-charts

# Update the repository
helm repo update

# Install DCGM Exporter
helm install \
    --generate-name \
    gpu-helm-charts/dcgm-exporter

# Apply DCGM Exporter YAML file to Kubernetes
kubectl create -f https://raw.githubusercontent.com/NVIDIA/dcgm-exporter/master/dcgm-exporter.yaml

# Get the name of the first DCGM Exporter pod
NAME=$(kubectl get pods -l "app.kubernetes.io/name=dcgm-exporter" \
                         -o "jsonpath={ .items[0].metadata.name}")

# Port forward the pod's 9400 port to local 8080
kubectl port-forward $NAME 8080:9400 &

# Access metrics at localhost:8080
curl -sL http://127.0.0.1:8080/metrics

# Fetch DCGM_FI_DEV_GPU_UTIL metric
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/services/dcgm-exporter-1744216530/DCGM_FI_DEV_GPU_UTIL" | jq .
```

---

### 4.3 Set Up Prometheus and Prometheus Adapter

```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community \
   https://prometheus-community.github.io/helm-charts

# Search for kube-prometheus in the repository
helm search repo kube-prometheus

# Export default values for kube-prometheus-stack
helm inspect values prometheus-community/kube-prometheus-stack > /tmp/kube-prometheus-stack.values

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

# Check for the DCGM_FI_DEV_MEM_COPY_UTIL metric
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq -r . | grep DCGM_FI_DEV_MEM_COPY_UTIL
```

---

### 4.4 Verify Horizontal Pod Autoscaler

Forward services to local ports for monitoring:
```bash
# Port forward Triton service to local port 8000
kubectl port-forward svc/triton-service 8000:8000 &

# Port forward Prometheus service to local port 9090
kubectl port-forward svc/kube-prometheus-stack-1744-prometheus -n prometheus 9090:9090 &

# Port forward DCGM Exporter service to local port 9400
kubectl port-forward svc/dcgm-exporter -n default 9400:9400 &
```




### 4.4 Configure Horizontal Pod Autoscaler (HPA)

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
        name: dcgm-exporter-1744216530 # Service name
      target:
        type: Value
        value: '2' # Target GPU utilization
```

Apply the HPA configuration:
```bash
kubectl apply -f hpa_gpu.yaml
```

Check the HPA status:
```bash
kubectl get hpa triton-hpa
```
