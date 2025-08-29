# Kubernetes Deployment for Transformer Models

This directory contains Kubernetes configuration files for deploying transformer models in a production environment.

## Configuration Files

1. **[namespace.yaml](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/environment/kubernetes/namespace.yaml)** - Creates the `transformer-demo` namespace
2. **[configmap.yaml](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/environment/kubernetes/configmap.yaml)** - Application configuration parameters
3. **[deployment.yaml](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/environment/kubernetes/deployment.yaml)** - Main deployment configuration
4. **[service.yaml](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/environment/kubernetes/service.yaml)** - Service to expose the deployment
5. **[hpa.yaml](file:///C:/Users/wahyu/Documents/Large-Language-Model-DEMO/environment/kubernetes/hpa.yaml)** - Horizontal Pod Autoscaler for automatic scaling

## Deployment Instructions

### Prerequisites
- Kubernetes cluster with GPU support
- kubectl configured to access your cluster
- Docker images built and available in a registry

### Deploying to Kubernetes

1. **Create the namespace:**
   ```bash
   kubectl apply -f namespace.yaml
   ```

2. **Apply the ConfigMap:**
   ```bash
   kubectl apply -f configmap.yaml
   ```

3. **Deploy the application:**
   ```bash
   kubectl apply -f deployment.yaml
   ```

4. **Expose the service:**
   ```bash
   kubectl apply -f service.yaml
   ```

5. **Enable auto-scaling:**
   ```bash
   kubectl apply -f hpa.yaml
   ```

### Or deploy everything at once:
```bash
kubectl apply -f .
```

## Monitoring and Management

### Check deployment status:
```bash
kubectl get deployments -n transformer-demo
```

### Check pods:
```bash
kubectl get pods -n transformer-demo
```

### Check services:
```bash
kubectl get services -n transformer-demo
```

### View logs:
```bash
kubectl logs -l app=transformer-api -n transformer-demo
```

### Scale manually (if needed):
```bash
kubectl scale deployment transformer-api --replicas=5 -n transformer-demo
```

## Configuration Details

### Resource Requirements
- **CPU**: 1-2 cores per pod
- **Memory**: 2-4 GB RAM per pod
- **GPU**: 1 NVIDIA GPU per pod (requires nvidia-docker runtime)

### Environment Variables
The deployment uses the following environment variables from the ConfigMap:
- `MAX_WORKERS`: Number of worker processes
- `TIMEOUT`: Request timeout in seconds
- `MODEL_TYPE`: Type of transformer model to load
- `MAX_LENGTH`: Maximum sequence length for generation
- `TEMPERATURE`: Sampling temperature for text generation
- `BATCH_SIZE`: Batch size for inference
- `CACHE_SIZE`: Size of the response cache
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARN, ERROR)
- `LOG_FORMAT`: Log format (json or text)

## GPU Support

The deployment is configured to use NVIDIA GPUs with the following settings:
- `nvidia.com/gpu: 1` resource request/limit
- `NVIDIA_VISIBLE_DEVICES: all` environment variable
- `NVIDIA_DRIVER_CAPABILITIES: compute,utility` environment variable
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
kubectl describe deployment transformer-api -n transformer-demo

# Get detailed pod information
kubectl describe pods -l app=transformer-api -n transformer-demo

# Check resource usage
kubectl top pods -n transformer-demo

# Port forward for local testing
kubectl port-forward service/transformer-api-service 8000:80 -n transformer-demo
```