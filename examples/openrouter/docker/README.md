# Docker Deployment Guide for GenOps OpenRouter Service

This directory contains production-ready Docker configurations for deploying the GenOps OpenRouter service with comprehensive AI governance capabilities.

## Quick Start

### 1. Setup Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
vim .env
```

Required environment variables:
```bash
OPENROUTER_API_KEY=your-openrouter-api-key-here
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io
OTEL_EXPORTER_OTLP_HEADERS=x-honeycomb-team=your-honeycomb-key
```

### 2. Build and Run

```bash
# Build the Docker image
docker build -t genops/openrouter-service .

# Run with Docker Compose (recommended)
docker-compose up -d

# Or run standalone
docker run -d \
  --name openrouter-service \
  --env-file .env \
  -p 8000:8000 \
  genops/openrouter-service
```

### 3. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Test API endpoint
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}],
    "team": "docker-test",
    "max_tokens": 50
  }'

# View logs
docker-compose logs -f openrouter-service
```

## Docker Compose Stack

The complete stack includes:

- **openrouter-service**: Main GenOps OpenRouter service
- **traefik**: Reverse proxy with SSL termination
- **jaeger**: Local distributed tracing (optional)
- **prometheus**: Metrics collection (optional)
- **grafana**: Monitoring dashboards (optional)

### Access URLs

- **API Service**: http://openrouter.localhost
- **Traefik Dashboard**: http://traefik.localhost:8080
- **Jaeger UI**: http://jaeger.localhost
- **Prometheus**: http://prometheus.localhost
- **Grafana**: http://grafana.localhost (admin/admin)

## Production Configuration

### Security Hardening

The Docker image includes production security features:

```dockerfile
# Non-root user
USER appuser

# Read-only filesystem
read_only: true

# No new privileges
security_opt:
  - no-new-privileges:true

# Drop all capabilities
cap_drop:
  - ALL
```

### Health Checks

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "from genops.providers.openrouter import validate_setup; exit(0 if validate_setup().is_valid else 1)"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
    reservations:
      memory: 512M
      cpus: '0.25'
```

## Observability Integration

### Local Stack (Development)

```bash
# Start full observability stack
docker-compose up -d jaeger prometheus grafana

# View traces in Jaeger
open http://jaeger.localhost

# View metrics in Grafana
open http://grafana.localhost
```

### Production Platforms

Update `.env` for your observability platform:

**Honeycomb:**
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io
OTEL_EXPORTER_OTLP_HEADERS=x-honeycomb-team=your-key
```

**Datadog:**
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.datadoghq.com
OTEL_EXPORTER_OTLP_HEADERS=dd-api-key=your-key
```

**New Relic:**
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.nr-data.net
OTEL_EXPORTER_OTLP_HEADERS=api-key=your-key
```

## API Usage Examples

### Basic Chat Completion

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "team": "engineering",
    "project": "ai-chatbot",
    "customer_id": "demo-001",
    "max_tokens": 100
  }'
```

### Cost Estimation

```bash
curl -X POST http://localhost:8000/cost/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-sonnet",
    "input_tokens": 200,
    "output_tokens": 100
  }'
```

### Multi-Provider Comparison

```bash
# Test different providers for cost comparison
for model in "openai/gpt-3.5-turbo" "anthropic/claude-3-haiku" "meta-llama/llama-3.2-3b-instruct"; do
  echo "Testing $model..."
  curl -s -X POST http://localhost:8000/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$model\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],
      \"team\": \"comparison\",
      \"max_tokens\": 20
    }" | jq '.usage.total_tokens'
done
```

## Scaling and Load Balancing

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy as stack
docker stack deploy -c docker-compose.yml genops

# Scale service
docker service scale genops_openrouter-service=3

# Update service
docker service update --image genops/openrouter-service:v1.1 genops_openrouter-service
```

### Multiple Instances

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  openrouter-service:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

## Monitoring and Logging

### Structured Logging

All logs are in JSON format for easy parsing:

```bash
# View logs with jq
docker-compose logs openrouter-service | jq -r .message

# Filter error logs
docker-compose logs openrouter-service | jq 'select(.level == "error")'

# Monitor real-time logs
docker-compose logs -f openrouter-service | jq .
```

### Metrics Collection

The service exposes Prometheus metrics:

```bash
# View metrics
curl http://localhost:8000/metrics

# Example metrics:
# - http_requests_total
# - http_request_duration_seconds
# - openrouter_requests_total
# - openrouter_cost_total
```

### Log Aggregation

For production, integrate with log aggregation:

```yaml
logging:
  driver: "fluentd"
  options:
    fluentd-address: "localhost:24224"
    tag: "openrouter.service"
```

## Backup and Recovery

### Configuration Backup

```bash
# Backup environment and compose files
tar -czf genops-openrouter-config.tar.gz .env docker-compose.yml

# Backup volumes
docker run --rm -v genops_app_data:/data -v $(pwd):/backup alpine \
  tar -czf /backup/app-data-backup.tar.gz -C /data .
```

### Restore Process

```bash
# Restore configuration
tar -xzf genops-openrouter-config.tar.gz

# Restore volumes
docker run --rm -v genops_app_data:/data -v $(pwd):/backup alpine \
  tar -xzf /backup/app-data-backup.tar.gz -C /data
```

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs openrouter-service

# Check environment variables
docker-compose exec openrouter-service env | grep OPENROUTER

# Validate configuration
docker-compose exec openrouter-service python -c "
from genops.providers.openrouter import validate_setup, print_validation_result
print_validation_result(validate_setup())
"
```

**API not responding:**
```bash
# Check container health
docker-compose ps

# Test internal connectivity
docker-compose exec openrouter-service curl -f http://localhost:8000/health

# Check port binding
netstat -tulpn | grep 8000
```

**High memory usage:**
```bash
# Monitor resource usage
docker stats openrouter-service

# Check for memory leaks
docker-compose exec openrouter-service python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'CPU: {psutil.cpu_percent()}%')
"
```

### Performance Tuning

**Gunicorn Configuration:**
```python
# In Dockerfile, adjust workers based on CPU cores
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--threads", "2", "--timeout", "120", "app:app"]
```

**Resource Optimization:**
```yaml
# In docker-compose.yml
services:
  openrouter-service:
    deploy:
      resources:
        limits:
          memory: 2G      # Increase for high load
          cpus: '1.0'     # Adjust based on usage
```

## Maintenance

### Updates

```bash
# Pull latest image
docker-compose pull openrouter-service

# Rolling update
docker-compose up -d --no-deps openrouter-service

# Verify update
curl http://localhost:8000/health
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (careful!)
docker-compose down -v

# Clean up unused images
docker image prune -a
```

## Production Checklist

- [ ] Environment variables configured
- [ ] API keys secured and rotated regularly
- [ ] SSL/TLS termination configured
- [ ] Resource limits set appropriately
- [ ] Health checks configured
- [ ] Logging configured for aggregation
- [ ] Monitoring and alerting set up
- [ ] Backup procedures tested
- [ ] Security scanning completed

## Support

- **Documentation**: [Full Integration Guide](../../docs/integrations/openrouter.md)
- **Examples**: [OpenRouter Examples](../)
- **Monitoring**: Check Grafana dashboards
- **Issues**: GitHub repository

---

**Production Ready**: This Docker configuration has been tested with production workloads and includes security hardening, monitoring, and scalability features.