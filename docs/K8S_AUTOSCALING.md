# Kubernetes GPU Autoscaling

Optional notes for running Nexus Forge API on Kubernetes with GPU-backed inference and horizontal scaling.

## Overview

The API serving layer uses `textSummarizer.serving.gpu_pool.GPUJobPool` to:

- Limit concurrent GPU inference (`MAX_GPU_JOBS`, default `2`)
- Cache loaded models with TTL eviction (`MODEL_CACHE_TTL_SECONDS`, default `3600`)

Pair this with Kubernetes HPA and GPU node pools for production autoscaling.

## Deployment sketch

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-forge-api
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: api
          image: nexus-forge:latest
          env:
            - name: MAX_GPU_JOBS
              value: "2"
            - name: MODEL_CACHE_TTL_SECONDS
              value: "3600"
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: nexus-forge-secrets
                  key: api-key
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              cpu: "2"
              memory: 8Gi
```

## Horizontal Pod Autoscaler

Scale on CPU or custom metrics (request rate, queue depth):

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nexus-forge-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nexus-forge-api
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

For GPU workloads, consider:

- **Cluster autoscaler** on GPU node groups (scale nodes when pods are pending)
- **KEDA** with Prometheus metrics from your ingress or API gateway
- **Per-pod `MAX_GPU_JOBS`** tuned to GPU memory (e.g. `1` for large models, `2` for BART)

## Node pool recommendations

| Workload | Suggested GPU | `MAX_GPU_JOBS` |
|----------|---------------|----------------|
| BART / FLAN-T5 small | T4 / L4 | 2 |
| Pegasus large | A10 / A100 | 1 |
| Extractive only | CPU nodes | N/A |

## Health checks

Use the existing `/health` endpoint for liveness and readiness probes. Warm the extractive model on startup via the FastAPI lifespan hook.

## Related docs

- [DEPLOY.md](DEPLOY.md) — Docker Compose and HF Spaces autoscaling
- [GEVAL.md](GEVAL.md) — G-Eval API key setup
