# 部署指南

## Docker

```bash
docker build -t dexworld .
docker run --rm dexworld python -m dexworld.cli doctor
```

## Docker Compose

```bash
docker-compose up -d
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

## Helm

```bash
helm install dexworld deploy/helm/dexworld
```

## HTTP 服务

```bash
python -m dexworld.cli serve --port 8765
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/metrics
```
