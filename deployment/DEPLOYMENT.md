# SpectraShield Deployment Guide

## Deployment Options

### 1. Docker Compose (Recommended for Development)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Kubernetes (Production)

#### Prerequisites
- Kubernetes cluster (EKS, GKE, AKS, or self-hosted)
- kubectl configured
- Docker images pushed to registry

#### Deploy

```bash
# Create namespace
kubectl create namespace spectrashield

# Apply configurations
kubectl apply -f deployment/kubernetes/deployments.yaml -n spectrashield
kubectl apply -f deployment/kubernetes/services.yaml -n spectrashield

# Check status
kubectl get pods -n spectrashield
kubectl get services -n spectrashield
```

#### Access Application

```bash
# Get frontend service URL
kubectl get service frontend-service -n spectrashield

# Port forward for local access
kubectl port-forward service/frontend-service 3000:80 -n spectrashield
```

### 3. AWS (Terraform)

#### Prerequisites
- AWS CLI configured
- Terraform installed
- AWS credentials set up

#### Deploy

```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan

# Apply infrastructure
terraform apply

# Get outputs
terraform output
```

#### Resources Created
- EKS Cluster
- ECR Repositories
- VPC and Networking
- Load Balancers
- Auto-scaling groups

### 4. Manual Deployment

#### Backend

```bash
cd backend

# Install dependencies
npm install --production

# Set environment variables
export PORT=4000
export DB_URI=mongodb://your-mongo-url
export REDIS_HOST=your-redis-host
export ML_ENGINE_URL=http://ml-engine:5000

# Start server
npm start
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Build
npm run build

# Start
npm start
```

#### ML Engine

```bash
cd ml-engine

# Install dependencies
pip install -r requirements.txt

# Start API server
python api.py
```

## Environment Configuration

### Production Environment Variables

**Backend**
```env
NODE_ENV=production
PORT=4000
DB_URI=mongodb://prod-mongo:27017/spectrashield
REDIS_HOST=prod-redis
REDIS_PORT=6379
ML_ENGINE_URL=http://ml-engine:5000
FRONTEND_URL=https://spectrashield.example.com
```

**Frontend**
```env
NEXT_PUBLIC_API_URL=https://api.spectrashield.example.com
```

## Monitoring

### Prometheus + Grafana

```bash
# Deploy monitoring stack
kubectl apply -f deployment/monitoring/prometheus.yaml
kubectl apply -f deployment/monitoring/grafana.yaml
```

### Logging

```bash
# Deploy ELK stack
kubectl apply -f deployment/monitoring/elasticsearch.yaml
kubectl apply -f deployment/monitoring/logstash.yaml
kubectl apply -f deployment/monitoring/kibana.yaml
```

## Scaling

### Horizontal Pod Autoscaling

```bash
# Backend autoscaling
kubectl autoscale deployment spectrashield-backend \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n spectrashield

# Frontend autoscaling
kubectl autoscale deployment spectrashield-frontend \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n spectrashield
```

### Database Scaling

**MongoDB**
- Use MongoDB Atlas for managed scaling
- Or deploy MongoDB replica set in Kubernetes

**Redis**
- Use Redis Cluster mode
- Or managed Redis (AWS ElastiCache, Azure Cache)

## Security

### SSL/TLS

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create certificate
kubectl apply -f deployment/kubernetes/certificate.yaml
```

### Secrets Management

```bash
# Create secrets
kubectl create secret generic backend-secrets \
  --from-literal=db-uri=mongodb://... \
  --from-literal=redis-password=... \
  -n spectrashield
```

## Backup and Recovery

### Database Backup

```bash
# MongoDB backup
mongodump --uri="mongodb://..." --out=/backup/$(date +%Y%m%d)

# Restore
mongorestore --uri="mongodb://..." /backup/20231119
```

### Blockchain Backup

```bash
# Backup blockchain data
kubectl cp spectrashield-backend-pod:/app/data/blockchain.json ./blockchain-backup.json
```

## Troubleshooting

### Check Logs

```bash
# Backend logs
kubectl logs -f deployment/spectrashield-backend -n spectrashield

# ML Engine logs
kubectl logs -f deployment/spectrashield-ml-engine -n spectrashield
```

### Debug Pods

```bash
# Get pod shell
kubectl exec -it spectrashield-backend-pod -n spectrashield -- /bin/sh

# Check connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
```

### Common Issues

**Issue**: ML Engine out of memory
**Solution**: Increase memory limits in deployment.yaml

**Issue**: Redis connection timeout
**Solution**: Check Redis service and network policies

**Issue**: MongoDB connection failed
**Solution**: Verify DB_URI and network connectivity

## Performance Optimization

### Caching
- Enable Redis caching for API responses
- Use CDN for frontend static assets

### Database Indexing
```javascript
// MongoDB indexes
db.analyses.createIndex({ status: 1, createdAt: -1 })
db.analyses.createIndex({ blockchainHash: 1 })
```

### Load Balancing
- Use Nginx or cloud load balancer
- Configure health checks

## Cost Optimization

### AWS
- Use Spot Instances for ML workloads
- Enable auto-scaling
- Use S3 for video storage

### Resource Limits
```yaml
resources:
  limits:
    memory: "2Gi"
    cpu: "1000m"
  requests:
    memory: "1Gi"
    cpu: "500m"
```

## Maintenance

### Updates

```bash
# Update backend
kubectl set image deployment/spectrashield-backend \
  backend=spectrashield-backend:v1.1.0 \
  -n spectrashield

# Rollback if needed
kubectl rollout undo deployment/spectrashield-backend -n spectrashield
```

### Health Checks

```bash
# Check all services
curl http://backend:4000/health
curl http://ml-engine:5000/health
```
