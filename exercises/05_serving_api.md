# Exercise 5: Serving a Model API

## Objective
Deploy a trained transformer model as a REST API service using FastAPI and containerize it with Docker.

## Prerequisites
- Completion of Exercises 1-4
- Understanding of model training and inference
- Basic knowledge of web frameworks and Docker

## Instructions

### Part 1: FastAPI Application

1. Create a FastAPI application:
   - Set up basic endpoints for health check and model info
   - Implement text generation endpoint
   - Add proper error handling and validation
   - Include API documentation with Pydantic models

2. Model loading and management:
   - Implement model loading on startup
   - Handle model initialization errors gracefully
   - Support multiple model variants (if applicable)
   - Implement model reloading without restarting the service

3. Request/response handling:
   - Define Pydantic models for API requests and responses
   - Implement input validation and sanitization
   - Add request logging and timing metrics
   - Handle batch requests for improved throughput

### Part 2: Performance Optimization

1. Inference optimization:
   - Implement caching for frequent requests
   - Add request batching for concurrent requests
   - Use torch.compile for model optimization (if applicable)
   - Implement mixed precision inference

2. Memory management:
   - Monitor GPU/CPU memory usage
   - Implement memory cleanup between requests
   - Add memory usage limits and alerts
   - Optimize model loading for faster startup

3. Concurrency handling:
   - Configure appropriate number of workers
   - Implement request queuing for high load
   - Add rate limiting to prevent abuse
   - Handle graceful shutdown of ongoing requests

### Part 3: Docker Containerization

1. Create Dockerfile:
   - Use appropriate base image (Python + CUDA support)
   - Install dependencies with proper versioning
   - Copy model files and application code
   - Set up proper user permissions
   - Expose necessary ports

2. Multi-stage build:
   - Implement multi-stage build for smaller image size
   - Separate build dependencies from runtime dependencies
   - Optimize layer caching for faster rebuilds

3. Docker Compose (optional):
   - Create docker-compose.yml for multi-service setup
   - Include monitoring and logging services
   - Add volume mounts for configuration and logs

### Part 4: Deployment and Monitoring

1. Kubernetes deployment:
   - Create Kubernetes deployment YAML
   - Configure resource requests and limits
   - Add health checks and readiness probes
   - Implement horizontal pod autoscaling

2. Monitoring and logging:
   - Add Prometheus metrics endpoint
   - Implement structured logging
   - Add application performance monitoring
   - Set up alerting for critical metrics

3. Load testing:
   - Create simple load testing script
   - Measure throughput and latency under load
   - Test error handling and recovery
   - Benchmark different batch sizes and configurations

### Part 5: Security and Reliability

1. Security considerations:
   - Implement API authentication and authorization
   - Add input validation to prevent injection attacks
   - Configure CORS policies appropriately
   - Secure environment variables and secrets

2. Reliability features:
   - Implement circuit breaker pattern
   - Add retry logic with exponential backoff
   - Implement graceful degradation for non-critical features
   - Add health check endpoints for load balancers

3. Backup and recovery:
   - Implement model checkpointing
   - Add configuration backup strategies
   - Create disaster recovery procedures
   - Test failover scenarios

## Challenge Problems

1. **Advanced Deployment**: Implement a more sophisticated deployment:
   - Use Kubernetes operators for model deployment
   - Implement blue-green deployment strategy
   - Add canary release capabilities
   - Implement auto-scaling based on custom metrics

2. **Performance Benchmarking**: Conduct comprehensive performance analysis:
   - Benchmark different hardware configurations (CPU vs GPU)
   - Test various optimization techniques
   - Measure latency percentiles under different loads
   - Compare synchronous vs asynchronous request handling

3. **Multi-Model Serving**: Extend the API to serve multiple models:
   - Implement model versioning
   - Add model routing based on request parameters
   - Implement model loading on demand
   - Add model-specific configuration management

## Submission

Create the following files:
- `api/main.py`: FastAPI application with all endpoints
- `api/models.py`: Pydantic models for request/response
- `Dockerfile`: Docker configuration for the application
- `docker-compose.yml` (optional): Multi-service setup
- `k8s/deployment.yaml`: Kubernetes deployment configuration
- `tests/load_test.py`: Simple load testing script
- `README.md`: Documentation with setup and usage instructions

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)