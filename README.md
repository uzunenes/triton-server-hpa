# Triton Server HPA

> GPU-based Horizontal Pod Autoscaling for NVIDIA Triton Inference Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kubernetes](https://img.shields.io/badge/kubernetes-1.30%2B-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Triton](https://img.shields.io/badge/Triton-25.06-76B900?logo=nvidia&logoColor=white)](https://github.com/triton-inference-server/server)

Build an AI inference service that grows and shrinks with demand. A YOLOv7-tiny
model is optimised with TensorRT, served by Triton on Kubernetes, and a
HorizontalPodAutoscaler adds replicas when **GPU utilisation** rises — not CPU,
not memory. One physical GPU is time-sliced so several Triton pods share it,
which makes the whole thing demonstrable on a single-GPU machine.

![Clients hit a Kubernetes service in front of a Triton deployment; DCGM exporter feeds GPU utilisation to Prometheus, the Prometheus adapter and the custom metrics API, which the HPA reads to scale the deployment.](docs/images/triton-server-hpa_architecture.svg?raw=true)

## What it looks like when it works

```text
TIME   GPU_UTIL   DESIRED   REPLICAS   READY
 10s   39         2         2          1
 40s   53         4         4          3
 60s   53         5         5          5
...load stops...
160s   0          4         4          4
280s   0          2         2          2
340s   0          1         1          1
```

```console
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
212072, tritonserver, 620 MiB
217232, tritonserver, 620 MiB
217812, tritonserver, 620 MiB
218325, tritonserver, 620 MiB
218827, tritonserver, 620 MiB      # five pods, one GPU
```

<details>
<summary><b>Verified on</b> (last full end-to-end run: 2026-09-10)</summary>

| Component | Version |
| --- | --- |
| GPU / driver | RTX 4060 Ti (sm_89) / 575.51.03 |
| OS | Ubuntu 22.04.5 LTS |
| minikube / Kubernetes | v1.39.0 / v1.37.0 |
| Triton Inference Server | `nvcr.io/nvidia/tritonserver:25.06-py3` |
| TensorRT (engine build) | `nvcr.io/nvidia/tensorrt:25.06-py3` (TensorRT 10.11) |
| NVIDIA device plugin | chart 0.20.0 |
| dcgm-exporter | chart 4.8.3 |
| kube-prometheus-stack | chart 70.x |
| Sustained load | 48 client threads, ~359 req/s |

</details>

---

## Table of Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [1. Cluster and GPU setup](#1-cluster-and-gpu-setup)
  - [1.1 NVIDIA Container Toolkit](#11-nvidia-container-toolkit)
  - [1.2 minikube](#12-minikube)
  - [1.3 kubectl](#13-kubectl)
  - [1.4 Helm](#14-helm)
  - [1.5 GPU time-slicing](#15-gpu-time-slicing)
- [2. Prepare the YOLOv7 model](#2-prepare-the-yolov7-model)
  - [2.1 Export to ONNX](#21-export-to-onnx)
  - [2.2 Build the TensorRT engine](#22-build-the-tensorrt-engine)
- [3. Deploy Triton Inference Server](#3-deploy-triton-inference-server)
- [4. Deploy the metrics pipeline](#4-deploy-the-metrics-pipeline)
  - [4.1 Prometheus](#41-prometheus)
  - [4.2 DCGM exporter](#42-dcgm-exporter)
  - [4.3 Prometheus Adapter](#43-prometheus-adapter)
- [5. Configure the HorizontalPodAutoscaler](#5-configure-the-horizontalpodautoscaler)
- [6. Generate load and watch it scale](#6-generate-load-and-watch-it-scale)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Why not the GPU Operator?](#why-not-the-gpu-operator)
- [References](#references)

---

## Requirements

- An NVIDIA GPU with a driver already installed on the host (`nvidia-smi` works).
- Docker, and enough disk for the container images (~40 GB).
- Around 16 GB of RAM.

> [!IMPORTANT]
> **Pick a container tag that matches your driver.** NGC containers refuse to
> start on an older driver, and GeForce cards do not get CUDA forward
> compatibility. A `26.08` container on a 575 driver fails with:
>
> ```text
> ERROR: This container was built for NVIDIA Driver Release 615.65 or later
> [[System has unsupported display driver / cuda driver combination
>   (CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) cuInit()=803]]
> ```
>
> Check your driver with `nvidia-smi --query-gpu=driver_version --format=csv`
> and pick the matching tag from the
> [NGC framework support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).
> This guide pins `25.06`, which needs driver 575 or newer.
>
> **The TensorRT tag and the Triton tag must be the same.** A serialized
> TensorRT engine only loads in the TensorRT version that built it.

---

## Quick start

If the prerequisites are already in place:

```bash
git clone https://github.com/uzunenes/triton-server-hpa.git
cd triton-server-hpa

# GPU time-slicing: advertise 10 shares of one physical GPU
kubectl label node minikube nvidia.com/gpu.present=true --overwrite
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin --create-namespace \
  -f helm-values/nvidia-device-plugin.values.yaml

# Triton (expects the engine at /mnt/triton_models/yolov7tiny/1/model.plan)
kubectl apply -f k8s/triton-deployment.yaml -f k8s/triton-service.yaml

# Metrics pipeline, then the HPA
helm upgrade -i kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --create-namespace --namespace prometheus \
  -f helm-values/kube-prometheus-stack.values.yaml --wait
helm upgrade -i dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace default -f helm-values/dcgm-exporter.values.yaml
helm upgrade -i prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace prometheus \
  --set prometheus.url=http://kube-prometheus-stack-prometheus.prometheus.svc \
  --set prometheus.port=9090
kubectl apply -f k8s/hpa-gpu.yaml
```

The sections below explain each step, including how to produce the model.

---

## 1. Cluster and GPU setup

### 1.1 NVIDIA Container Toolkit

Lets Docker containers use the GPU.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

> [!NOTE]
> Older versions of this guide ran
> `sed -i -e '/experimental/ s/^#//g' …nvidia-container-toolkit.list`, which
> uncomments NVIDIA's **experimental** apt channel. That channel ships release
> candidates (`1.20.0~rc.1`), so apt can pull a pre-release toolkit. The stable
> channel above is enough — leave the experimental line commented out.

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

`--gpus all` works as soon as the toolkit is installed; you do not need to edit
`/etc/docker/daemon.json` for this check.

---

### 1.2 minikube

```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

# Create the model repository BEFORE starting, or the mount has nothing to bind to
sudo mkdir -p /mnt/triton_models/yolov7tiny/1

minikube start --driver docker --container-runtime docker --gpus all --force \
  --mount --mount-string="/mnt/triton_models:/mnt/triton_models"
```

> [!WARNING]
> `--gpus all` is only accepted together with `--container-runtime docker`.
> Passing `--container-runtime containerd` fails with
> `The gpus flag can only be used with the docker driver and docker container-runtime`.
> This constraint is the reason this guide does not use the GPU Operator —
> see [Why not the GPU Operator?](#why-not-the-gpu-operator).

Verify:

```bash
minikube status
minikube ssh -- ls /mnt/triton_models   # the mount is visible inside the node
```

---

### 1.3 kubectl

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && rm kubectl
kubectl get nodes
```

---

### 1.4 Helm

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh && ./get_helm.sh
```

---

### 1.5 GPU time-slicing

`minikube --gpus all` enables its own `nvidia-device-plugin` addon, which
advertises exactly **one** `nvidia.com/gpu`. One resource means one Triton pod,
so the autoscaler would have nothing to scale into. Replace it with the
standalone device plugin configured for time-slicing:

```bash
# The bundled addon advertises 1 GPU; swap it out
minikube addons disable nvidia-device-plugin

# The chart's node affinity needs one of the NFD labels. minikube has none,
# so set the label by hand.
kubectl label node minikube nvidia.com/gpu.present=true --overwrite

helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin --create-namespace \
  -f helm-values/nvidia-device-plugin.values.yaml
```

[`helm-values/nvidia-device-plugin.values.yaml`](helm-values/nvidia-device-plugin.values.yaml)
sets `replicas: 10`, so one physical GPU is advertised as ten schedulable units.

Verify — this must print `10`:

```bash
kubectl get node minikube -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'
```

Optionally run a test pod:

```bash
kubectl apply -f k8s/cuda-test-pod.yaml
kubectl logs gpu-test
kubectl delete -f k8s/cuda-test-pod.yaml
```

---

## 2. Prepare the YOLOv7 model

### 2.1 Export to ONNX

```bash
git clone --depth 1 https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

python3 -m venv .venv
.venv/bin/pip install -r ../requirements-export.txt

.venv/bin/python export.py --weights ./yolov7-tiny.pt \
  --grid --end2end --dynamic-batch --simplify \
  --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
```

> [!IMPORTANT]
> [`requirements-export.txt`](requirements-export.txt) pins **`torch==2.5.1` on
> purpose.** YOLOv7's `export.py` does not work with current PyTorch:
>
> - PyTorch **2.6** flipped `torch.load()` to `weights_only=True`, so the export
>   dies with `_pickle.UnpicklingError: Weights only load failed …
>   Unsupported global: GLOBAL models.yolo.Model`.
> - PyTorch **2.9+** routes `torch.onnx.export` through the dynamo exporter,
>   which needs `onnxscript` and does not emit the end2end graph. The run then
>   *appears* to succeed while writing no `.onnx` file at all.
>
> If you must use a newer torch, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` gets past
> the first problem but not the second.

`--end2end` bakes the TensorRT `EfficientNMS` plugin into the graph, so **NMS
runs on the GPU inside Triton** and the client receives final boxes.

### 2.2 Build the TensorRT engine

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work \
  nvcr.io/nvidia/tensorrt:25.06-py3 \
  trtexec --onnx=./yolov7-tiny.onnx \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:8x3x640x640 \
    --maxShapes=images:8x3x640x640 \
    --fp16 --memPoolSize=workspace:4096 \
    --saveEngine=model.plan --timingCacheFile=timing.cache

sudo cp model.plan /mnt/triton_models/yolov7tiny/1/model.plan
```

> [!NOTE]
> Use `--memPoolSize=workspace:4096`. The older `--workspace=4096` flag still
> parses but prints `[Deprecated] this knob has been deprecated`.

The bind mount replaces the `docker run -it` + `docker cp <container_id>` dance
from earlier versions of this guide.

---

## 3. Deploy Triton Inference Server

```bash
kubectl apply -f k8s/triton-deployment.yaml -f k8s/triton-service.yaml
kubectl rollout status deploy/triton-inference-server
```

[`k8s/triton-deployment.yaml`](k8s/triton-deployment.yaml) requests
`nvidia.com/gpu: 1` — one time-slice, not one whole card — and carries
startup/readiness/liveness probes on Triton's `/v2/health/*` endpoints so
Kubernetes only sends traffic to a pod once its model is loaded.

Verify the model:

```bash
curl -s "http://$(minikube ip):30001/v2/models/yolov7tiny" | jq .
```

```json
{
  "name": "yolov7tiny",
  "versions": ["1"],
  "platform": "tensorrt_plan",
  "inputs":  [{"name": "images", "datatype": "FP32", "shape": [-1, 3, 640, 640]}],
  "outputs": [
    {"name": "num_dets",    "datatype": "INT32", "shape": [-1, 1]},
    {"name": "det_boxes",   "datatype": "FP32",  "shape": [-1, 100, 4]},
    {"name": "det_scores",  "datatype": "FP32",  "shape": [-1, 100]},
    {"name": "det_classes", "datatype": "INT32", "shape": [-1, 100]}
  ]
}
```

Run one inference:

```bash
pip install -r requirements.txt
python3 inference.py --mode detect --url "$(minikube ip):30001"
```

```text
2 detection(s) above 0.6
  person          0.906  [33 87 366 739]
  dog             0.818  [651 663 955 896]
wrote detection_result.jpg
```

![YOLOv7-tiny detections returned by Triton: a person at 0.91 and a dog at 0.82.](docs/images/detection_result.jpg?raw=true)

---

## 4. Deploy the metrics pipeline

The chain the autoscaler depends on:

```text
dcgm-exporter  ──ServiceMonitor──▶  Prometheus  ──▶  prometheus-adapter
       │                                                     │
  DCGM_FI_DEV_GPU_UTIL                          custom.metrics.k8s.io  ──▶  HPA
```

> [!IMPORTANT]
> **Install Prometheus first.** dcgm-exporter creates a `ServiceMonitor`, and
> that CRD ships with kube-prometheus-stack. Installing dcgm-exporter first
> fails with:
> `no matches for kind "ServiceMonitor" in version "monitoring.coreos.com/v1" — ensure CRDs are installed first`.

### 4.1 Prometheus

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade -i kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --create-namespace --namespace prometheus \
  -f helm-values/kube-prometheus-stack.values.yaml \
  --timeout 10m --wait
```

[`helm-values/kube-prometheus-stack.values.yaml`](helm-values/kube-prometheus-stack.values.yaml)
is a **23-line override**, not a copy of the chart's defaults. The setting that
matters is `serviceMonitorSelectorNilUsesHelmValues: false`; without it
Prometheus only selects ServiceMonitors it created itself and never scrapes
dcgm-exporter.

Using a fixed release name keeps the service name predictable —
`kube-prometheus-stack-prometheus` — so nothing downstream needs editing.

### 4.2 DCGM exporter

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update

helm upgrade -i dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace default -f helm-values/dcgm-exporter.values.yaml
```

Install it **once**, with a fixed release name, into the same namespace as the
Deployment the HPA scales. You do not need `datacenter-gpu-manager` on the host;
the exporter ships its own DCGM.

Verify the scrape target is healthy:

```bash
kubectl port-forward -n prometheus svc/kube-prometheus-stack-prometheus 9090:9090 &
curl -s 'http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL' \
  | jq -c '.data.result[0].metric'
```

```json
{"__name__":"DCGM_FI_DEV_GPU_UTIL","job":"dcgm-exporter","namespace":"default",
 "pod":"dcgm-exporter-blhff","service":"dcgm-exporter",
 "exported_pod":"triton-inference-server-7d4744756c-vr2fr", "...": "..."}
```

The `service` and `namespace` labels are the important part — they are what lets
the HPA ask about `Service/dcgm-exporter`, and **only a ServiceMonitor adds
them**. A hand-written `additionalScrapeConfigs` job (as in earlier versions of
this guide) attaches no such labels, so the HPA query cannot resolve.

### 4.3 Prometheus Adapter

```bash
helm upgrade -i prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace prometheus \
  --set rbac.create=true \
  --set prometheus.url=http://kube-prometheus-stack-prometheus.prometheus.svc \
  --set prometheus.port=9090
```

Verify the exact query the HPA will make:

```bash
kubectl get --raw \
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/services/dcgm-exporter/DCGM_FI_DEV_GPU_UTIL" | jq .
```

---

## 5. Configure the HorizontalPodAutoscaler

```bash
kubectl apply -f k8s/hpa-gpu.yaml
kubectl get hpa triton-hpa
```

```text
NAME         REFERENCE                            TARGETS   MINPODS   MAXPODS   REPLICAS
triton-hpa   Deployment/triton-inference-server   0/30      1         5         1
```

[`k8s/hpa-gpu.yaml`](k8s/hpa-gpu.yaml) targets **30 % GPU utilisation** and
allows 1–5 replicas. `DCGM_FI_DEV_GPU_UTIL` is a percentage per physical GPU, so
`30` reads as "start adding replicas once the card is 30 % busy".

`behavior` makes the demo legible: scale up immediately, two pods at a time;
scale down one pod per minute after a two-minute stabilisation window, so the
cluster does not oscillate.

---

## 6. Generate load and watch it scale

In one terminal:

```bash
python3 inference.py --mode load --url "$(minikube ip):30001" --threads 48 --duration 420
```

In another:

```bash
watch -n 5 'kubectl get hpa triton-hpa; kubectl get pods -l app=triton-inference-server; nvidia-smi'
```

Observed run: GPU utilisation crossed 30 % within 10 s, the deployment reached
five replicas after ~60 s, sustained **~359 req/s**, and returned to a single
replica about six minutes after the load stopped.

![kubectl and nvidia-smi during the load test: five Triton pods Running, GPU 47% busy with five tritonserver processes on GPU 0, HPA showing 47/30 and 5 replicas.](docs/images/result.jpg?raw=true)

---

## Repository layout

```text
k8s/                                    kubectl apply -f k8s/
  triton-deployment.yaml                Triton, 1 GPU slice, health probes
  triton-service.yaml                   NodePort 30001/30002/30003
  hpa-gpu.yaml                          HPA on DCGM_FI_DEV_GPU_UTIL
  cuda-test-pod.yaml                    quick GPU sanity check
helm-values/
  nvidia-device-plugin.values.yaml      time-slicing, replicas: 10
  dcgm-exporter.values.yaml             ServiceMonitor + node selector
  kube-prometheus-stack.values.yaml     23-line override
inference.py                            Triton client: detect / load modes
requirements.txt                        client dependencies
requirements-export.txt                 ONNX export dependencies (pinned torch)
```

---

## Troubleshooting

<details>
<summary><code>allocatable nvidia.com/gpu</code> is 1 instead of 10</summary>

minikube's bundled addon is still running, or the standalone plugin never
scheduled. Check both:

```bash
minikube addons list | grep nvidia-device-plugin      # should be disabled
kubectl get ds -n nvidia-device-plugin                # DESIRED should be 1
```

`DESIRED 0` means the DaemonSet's node affinity matched nothing. The chart
requires one of `nvidia.com/gpu.present=true`,
`feature.node.kubernetes.io/pci-10de.present=true`, or
`feature.node.kubernetes.io/cpu-model.vendor_id=NVIDIA`. Apply the first:

```bash
kubectl label node minikube nvidia.com/gpu.present=true --overwrite
```
</details>

<details>
<summary><code>Failed to create pod sandbox: RuntimeHandler "nvidia" not supported</code></summary>

Something installed a pod with `runtimeClassName: nvidia` — almost always the
GPU Operator. minikube's `--container-runtime docker` uses cri-dockerd, which
does not implement RuntimeClass handlers, so those pods can never start.
Uninstall the operator and use the standalone device plugin from
[step 1.5](#15-gpu-time-slicing):

```bash
helm uninstall gpu-operator -n default
```
</details>

<details>
<summary><code>MountVolume.SetUp failed … configmap "time-slicing-config" not found</code></summary>

A device plugin was pointed at a ConfigMap that does not exist yet, and the pod
stays in `Init:0/2` forever. Create the ConfigMap **before** the chart that
references it, or use `helm-values/nvidia-device-plugin.values.yaml`, which
inlines the config so there is no ordering problem.
</details>

<details>
<summary>HPA shows <code>&lt;unknown&gt;</code> or the adapter returns <code>the server could not find the metric</code></summary>

Walk the chain outward:

```bash
# 1. exporter emits it
kubectl port-forward svc/dcgm-exporter 9400:9400 &
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL

# 2. Prometheus scrapes it
curl -s 'http://localhost:9090/api/v1/targets?state=active' \
  | jq -r '.data.activeTargets[] | select(.scrapePool|test("dcgm")) | "\(.health) \(.lastError)"'

# 3. adapter exposes it
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq -r '.resources[].name' | grep DCGM
```

If step 2 lists no dcgm pool at all, prometheus-operator rejected the
ServiceMonitor. The usual cause is `scrapeTimeout` greater than `interval` —
the chart defaults to `scrapeTimeout: 25s`, so overriding only `interval: 5s`
silently drops the whole ServiceMonitor. Set both.
</details>

<details>
<summary><code>CUDA_ERROR_SYSTEM_DRIVER_MISMATCH</code> / <code>cuInit()=803</code></summary>

The container is newer than the host driver. See
[Requirements](#requirements) and pick a tag your driver supports.
</details>

<details>
<summary>Triton starts but reports no models</summary>

`/mnt/triton_models` was created after `minikube start`, so the 9p mount points
at an empty directory. Create the directory first, then restart:

```bash
sudo mkdir -p /mnt/triton_models/yolov7tiny/1
minikube stop && minikube start --driver docker --container-runtime docker \
  --gpus all --force --mount --mount-string="/mnt/triton_models:/mnt/triton_models"
```
</details>

---

## Why not the GPU Operator?

Earlier versions of this guide installed the NVIDIA GPU Operator. On minikube
that no longer works, and the reason is a hard conflict rather than a
misconfiguration:

1. minikube accepts `--gpus all` **only** with `--container-runtime docker`,
   which means cri-dockerd.
2. cri-dockerd does not implement RuntimeClass handlers.
3. Every current GPU Operator operand (validator, device plugin, dcgm-exporter,
   gpu-feature-discovery) sets `runtimeClassName: nvidia`.

The result is that all of them sit in `Init:0/N` forever with
`RuntimeHandler "nvidia" not supported`. On top of that,
`--set operator.defaultRuntime=docker` no longer exists in the chart, so Helm
accepts it and silently ignores it.

The GPU Operator is not needed here anyway. The driver is on the host, minikube
wires up the container toolkit, and dcgm-exporter is installed on its own — so
the operator's only remaining job is the device plugin, which the standalone
chart does without RuntimeClass. If you are on a real cluster with containerd,
the GPU Operator remains the right choice.

---

## References

- [NVIDIA — Scaling inference workloads](https://docs.nvidia.com/ai-enterprise/deployment/natural-language-processing/latest/scaling.html)
- [Kubernetes — Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [NVIDIA k8s-device-plugin — time-slicing](https://github.com/NVIDIA/k8s-device-plugin#shared-access-to-gpus)
- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [prometheus-adapter — walkthrough](https://github.com/kubernetes-sigs/prometheus-adapter/blob/master/docs/walkthrough.md)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [NGC framework support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
