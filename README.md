# Triton Server HPA with Go Requests

This repository demonstrates a **Horizontal Pod Autoscaler (HPA)** test setup using **Rancher**, **Kubernetes**, and **Triton Inference Server**.  
It showcases how to dynamically scale Triton Inference Server pods based on incoming traffic or resource usage. The test is powered by **Go**, where a custom Go client sends concurrent inference requests to the Triton server, simulating a real-world AI inference workload.

## Features

- Horizontal Pod Autoscaling (HPA) for Triton Inference Server.
- Integration with **Rancher** for Kubernetes cluster management.
- Custom **Go client** to send concurrent inference requests.
- Dynamically scales pods based on request load or resource usage.

## Prerequisites

- Kubernetes cluster (managed via Rancher or standalone).
- Docker and Kubernetes installed on the host machine.
- RTX 2080 Ti or compatible GPU for inference.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/triton-server-hpa.git
   cd triton-server-hpa
