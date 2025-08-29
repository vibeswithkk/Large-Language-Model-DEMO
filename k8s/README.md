# Kubernetes Deployment for LLM Demo API

This directory contains Kubernetes configuration files for deploying the LLM Demo API service in a production environment.

## Configuration Files

1. **[deployment.yaml](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/k8s/deployment.yaml)** - Complete deployment configuration including:
   - Deployment with 3 replicas
   - Service for internal and external access
   - Horizontal Pod Autoscaler for automatic scaling
   - ConfigMap for application configuration

## Deployment Instructions

### Prerequisites
- Kubernetes cluster with GPU support
- kubectl configured to access your cluster
- Docker images built and available in a registry

### Deploying to Kubernetes

1. **Apply the deployment**:
   ```bash
   kubectl apply -f deployment.yaml
   ```

### Or deploy step by step:
```bash
# Create the namespace
kubectl apply -f deployment.yaml --namespace llm-demo

# Deploy the application
kubectl apply -f deployment.yaml

# Check deployment status
kubectl get deployments -n llm-demo

# Check pods
kubectl get pods -n llm-demo

# Check services
kubectl get services -n llm-demo
```

## Monitoring and Management

### Check deployment status:
```bash
kubectl get deployments -n llm-demo
```

### Check pods:
```bash
kubectl get pods -n llm-demo
```

### Check services:
```bash
kubectl get services -n llm-demo
```

### View logs:
```bash
kubectl logs -l app=llm-demo-api -n llm-demo
```

### Scale manually (if needed):
```bash
kubectl scale deployment llm-demo-api --replicas=5 -n llm-demo
```

## Configuration Details

### Resource Requirements
- **CPU**: 2-4 cores per pod
- **Memory**: 4-8 GB RAM per pod
- **GPU**: 1 NVIDIA GPU per pod (requires nvidia-docker runtime)

### Environment Variables
The deployment uses the following environment variables:
- `MAX_WORKERS`: Number of worker processes
- `TIMEOUT`: Request timeout in seconds
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARN, ERROR)
- `MODEL_TYPE`: Type of model to load

## GPU Support

The deployment is configured to use NVIDIA GPUs with the following settings:
- `nvidia.com/gpu: 1` resource request/limit
- Toleration for GPU nodes

## Auto-scaling

The Horizontal Pod Autoscaler is configured to:
- Maintain between 2-10 replicas
- Scale based on CPU utilization (target 70%)
- Scale based on memory utilization (target 80%)

## Troubleshooting

### Common Issues

1. **Insufficient GPU resources:**
   - Ensure GPU nodes are available in the cluster
   - Check NVIDIA device plugin is installed

2. **Image pull errors:**
   - Verify the image name and tag
   - Ensure the image is accessible from the cluster

3. **Pods stuck in Pending:**
   - Check resource quotas
   - Verify node affinity/anti-affinity rules

### Useful Commands

```bash
# Describe deployment for detailed status
kubectl describe deployment llm-demo-api -n llm-demo

# Get detailed pod information
kubectl describe pods -l app=llm-demo-api -n llm-demo

# Check resource usage
kubectl top pods -n llm-demo

# Port forward for local testing
kubectl port-forward service/llm-demo-api-service 8000:80 -n llm-demo
```