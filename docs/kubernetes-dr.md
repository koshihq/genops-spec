# Disaster Recovery for GenOps AI on Kubernetes

> **Status:** 📋 Documentation in progress
> **Last Updated:** 2026-01-18

Build resilient GenOps AI deployments with comprehensive disaster recovery strategies and business continuity planning.

---

## Overview

Disaster recovery ensures your AI workloads can survive and recover from catastrophic failures:
- **Backup and Restore** of cluster state, configurations, and persistent data
- **High Availability** with multi-zone and multi-region deployments
- **Failover Automation** for rapid recovery with minimal downtime
- **Data Replication** across availability zones and regions
- **Recovery Testing** with regular DR drills and validation

GenOps AI's governance tracking continues across DR scenarios, ensuring cost attribution and compliance during recovery operations.

---

## Quick Reference

### Key DR Metrics

**Recovery Time Objective (RTO):**
- Maximum acceptable downtime
- Target: < 15 minutes for critical AI services

**Recovery Point Objective (RPO):**
- Maximum acceptable data loss
- Target: < 5 minutes for transaction data

**Service Level Objectives (SLO):**
- Target availability: 99.9% (8.76 hours downtime/year)
- Target MTTR: < 30 minutes

### DR Strategy Selection

| Strategy | RTO | RPO | Cost | Use Case |
|----------|-----|-----|------|----------|
| **Backup/Restore** | Hours | Hours | Low | Dev/Staging |
| **Pilot Light** | Minutes-Hours | Minutes | Medium | Non-critical production |
| **Warm Standby** | Minutes | Seconds | High | Business-critical |
| **Hot Standby (Active-Active)** | Seconds | None | Very High | Mission-critical |

---

## Table of Contents

### Planned Documentation Sections

1. **DR Strategy and Planning**
   - RTO/RPO definition and analysis
   - Business impact assessment
   - DR strategy selection
   - Cost-benefit analysis
   - Compliance requirements (SOC2, HIPAA, etc.)

2. **Backup Solutions**
   - Velero for cluster backup and restore
   - etcd backup strategies
   - Persistent volume snapshots
   - Configuration backup automation
   - Secrets and certificate backup

3. **High Availability Architecture**
   - Multi-zone deployments
   - Multi-region architectures
   - Cross-cluster service mesh
   - Database replication strategies
   - Stateless vs stateful service design

4. **Failover Automation**
   - Health checks and readiness probes
   - Automatic failover with DNS/load balancers
   - Traffic shifting strategies
   - Stateful application failover
   - Session persistence across failures

5. **Data Replication**
   - Persistent volume replication
   - Database replication (PostgreSQL, MongoDB, etc.)
   - S3/object storage cross-region replication
   - Conflict resolution strategies
   - Consistency guarantees

6. **Recovery Procedures**
   - Incident response runbooks
   - Cluster recovery from backup
   - Application restoration procedures
   - Data validation after recovery
   - Post-incident review process

7. **Testing and Validation**
   - DR drill planning and execution
   - Chaos engineering practices
   - Automated recovery testing
   - Performance validation post-recovery
   - Documentation and lessons learned

---

## Related Documentation

**Kubernetes Guides:**
- [Kubernetes Getting Started](kubernetes-getting-started.md)
- [Multi-Cloud Deployment](kubernetes-multi-cloud.md)
- [Best Practices](kubernetes-best-practices.md)

**High Availability:**
- [AWS Deployment Guide](kubernetes-aws-deployment.md)
- [Azure Deployment Guide](kubernetes-azure-deployment.md)
- [GCP Deployment Guide](kubernetes-gcp-deployment.md)

---

## Quick Examples

### Example 1: Velero Backup Configuration

```bash
# Install Velero with S3 backend
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.7.0 \
  --bucket genops-backup \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1 \
  --secret-file ./credentials-velero

# Create scheduled backup of GenOps AI namespace
velero schedule create genops-daily \
  --schedule="0 2 * * *" \
  --include-namespaces genops \
  --ttl 720h0m0s

# Backup specific resources with labels
velero backup create genops-manual \
  --selector app=genops-ai \
  --include-namespaces genops \
  --wait

# Restore from backup
velero restore create --from-backup genops-daily-20260118020000
```

### Example 2: Multi-Zone High Availability Deployment

```yaml
# Multi-zone deployment with pod topology spread
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai
  namespace: genops
spec:
  replicas: 6  # Spread across 3 zones
  selector:
    matchLabels:
      app: genops-ai
  template:
    metadata:
      labels:
        app: genops-ai
    spec:
      # Spread pods evenly across zones
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: genops-ai

      # Anti-affinity to avoid node collocation
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - genops-ai
            topologyKey: kubernetes.io/hostname

      # Pod disruption budget
      containers:
      - name: genops-ai
        image: genops-ai:latest
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"

        # Health checks for automatic recovery
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

---
# Pod Disruption Budget to maintain availability during updates
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: genops-ai-pdb
  namespace: genops
spec:
  minAvailable: 3  # At least 3 pods always running
  selector:
    matchLabels:
      app: genops-ai
```

### Example 3: Cross-Region Active-Passive Setup

```yaml
# Primary region deployment (active)
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-primary
  namespace: genops
  annotations:
    external-dns.alpha.kubernetes.io/hostname: api-primary.genops.example.com
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  selector:
    app: genops-ai
    region: us-east-1
  ports:
  - port: 443
    targetPort: 8080

---
# Secondary region deployment (passive standby)
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-secondary
  namespace: genops
  annotations:
    external-dns.alpha.kubernetes.io/hostname: api-secondary.genops.example.com
spec:
  type: LoadBalancer
  selector:
    app: genops-ai
    region: us-west-2
  ports:
  - port: 443
    targetPort: 8080

---
# Route53 health check and failover (AWS)
# Configured to route to primary, fail over to secondary if unhealthy
apiVersion: v1
kind: ConfigMap
metadata:
  name: dns-failover-config
  namespace: genops
data:
  route53-config.yaml: |
    primary:
      endpoint: api-primary.genops.example.com
      health_check_interval: 30s
      failure_threshold: 3
    secondary:
      endpoint: api-secondary.genops.example.com
      enabled_on_primary_failure: true
```

### Example 4: Automated etcd Backup

```yaml
# CronJob for etcd backup
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  successfulJobsHistoryLimit: 5
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: etcd-backup
          containers:
          - name: backup
            image: registry.k8s.io/etcd:3.5.9-0
            command:
            - /bin/sh
            - -c
            - |
              TIMESTAMP=$(date +%Y%m%d-%H%M%S)
              BACKUP_FILE="/backup/etcd-snapshot-${TIMESTAMP}.db"

              # Create etcd snapshot
              ETCDCTL_API=3 etcdctl snapshot save ${BACKUP_FILE} \
                --endpoints=https://etcd:2379 \
                --cacert=/etc/kubernetes/pki/etcd/ca.crt \
                --cert=/etc/kubernetes/pki/etcd/server.crt \
                --key=/etc/kubernetes/pki/etcd/server.key

              # Upload to S3
              aws s3 cp ${BACKUP_FILE} s3://genops-etcd-backup/

              # Cleanup old local backups
              find /backup -name "etcd-snapshot-*.db" -mtime +7 -delete

              echo "Backup completed: ${BACKUP_FILE}"
            env:
            - name: AWS_REGION
              value: us-east-1
            volumeMounts:
            - name: backup
              mountPath: /backup
            - name: etcd-certs
              mountPath: /etc/kubernetes/pki/etcd
              readOnly: true
          restartPolicy: OnFailure
          volumes:
          - name: backup
            persistentVolumeClaim:
              claimName: etcd-backup-pvc
          - name: etcd-certs
            hostPath:
              path: /etc/kubernetes/pki/etcd
              type: Directory
```

### Example 5: Persistent Volume Snapshot

```yaml
# VolumeSnapshotClass for AWS EBS
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshotClass
metadata:
  name: ebs-snapshot-class
driver: ebs.csi.aws.com
deletionPolicy: Retain
parameters:
  tagSpecification_1: "purpose=genops-backup"

---
# Create snapshot of PVC
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: genops-data-snapshot
  namespace: genops
spec:
  volumeSnapshotClassName: ebs-snapshot-class
  source:
    persistentVolumeClaimName: genops-data-pvc

---
# CronJob to create regular snapshots
apiVersion: batch/v1
kind: CronJob
metadata:
  name: genops-snapshot
  namespace: genops
spec:
  schedule: "0 1 * * *"  # Daily at 1 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: snapshot-creator
          containers:
          - name: create-snapshot
            image: bitnami/kubectl:latest
            command:
            - /bin/bash
            - -c
            - |
              TIMESTAMP=$(date +%Y%m%d-%H%M%S)

              kubectl apply -f - <<EOF
              apiVersion: snapshot.storage.k8s.io/v1
              kind: VolumeSnapshot
              metadata:
                name: genops-data-${TIMESTAMP}
                namespace: genops
              spec:
                volumeSnapshotClassName: ebs-snapshot-class
                source:
                  persistentVolumeClaimName: genops-data-pvc
              EOF

              echo "Snapshot created: genops-data-${TIMESTAMP}"
          restartPolicy: OnFailure
```

### Example 6: DR Runbook as Code

```yaml
# ConfigMap containing DR runbook
apiVersion: v1
kind: ConfigMap
metadata:
  name: dr-runbook
  namespace: genops
data:
  disaster-recovery.md: |
    # GenOps AI Disaster Recovery Runbook

    ## Incident Classification
    - **P0 (Critical)**: Complete service outage, RTO < 15 min
    - **P1 (High)**: Partial outage, degraded performance, RTO < 1 hour
    - **P2 (Medium)**: Non-critical service impact, RTO < 4 hours

    ## Recovery Procedures

    ### Scenario 1: Complete Cluster Failure

    1. **Assess Impact**
       ```bash
       # Check cluster health
       kubectl cluster-info
       kubectl get nodes
       ```

    2. **Activate Secondary Region**
       ```bash
       # Update DNS to point to secondary region
       aws route53 change-resource-record-sets \
         --hosted-zone-id Z1234567890ABC \
         --change-batch file://failover-to-secondary.json
       ```

    3. **Restore from Backup (if needed)**
       ```bash
       # List available backups
       velero backup get

       # Restore from latest backup
       velero restore create --from-backup genops-daily-latest
       ```

    4. **Verify Service Health**
       ```bash
       # Check all pods are running
       kubectl get pods -n genops

       # Test API endpoint
       curl -f https://api.genops.example.com/health
       ```

    ### Scenario 2: Data Corruption

    1. **Identify Affected Data**
    2. **Restore from Volume Snapshot**
    3. **Validate Data Integrity**
    4. **Resume Operations**

    ### Post-Recovery Checklist
    - [ ] All services restored and healthy
    - [ ] Data integrity validated
    - [ ] Governance tracking resumed (GenOps telemetry flowing)
    - [ ] Cost attribution accurate
    - [ ] Monitoring and alerting functional
    - [ ] Incident report filed
    - [ ] DR procedures updated
```

---

## DR Strategy and Planning (Comprehensive)

### RTO/RPO Analysis Framework

**Define Recovery Objectives:**
```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ServiceDRRequirements:
    """DR requirements for a service."""
    service_name: str
    rto_minutes: int  # Recovery Time Objective
    rpo_minutes: int  # Recovery Point Objective
    criticality: str  # critical, high, medium, low
    dependencies: List[str]
    data_volume_gb: float
    compliance_requirements: List[str]

# Example DR requirements mapping
DR_REQUIREMENTS = {
    "ai-inference-api": ServiceDRRequirements(
        service_name="ai-inference-api",
        rto_minutes=15,
        rpo_minutes=5,
        criticality="critical",
        dependencies=["redis-cache", "postgresql-db"],
        data_volume_gb=100.0,
        compliance_requirements=["SOC2", "HIPAA"]
    ),
    "batch-processing": ServiceDRRequirements(
        service_name="batch-processing",
        rto_minutes=240,
        rpo_minutes=60,
        criticality="medium",
        dependencies=["s3-storage"],
        data_volume_gb=1000.0,
        compliance_requirements=[]
    )
}

def calculate_dr_strategy(requirements: ServiceDRRequirements) -> str:
    """Determine appropriate DR strategy based on requirements."""
    if requirements.rto_minutes <= 15 and requirements.rpo_minutes <= 5:
        return "active-active"
    elif requirements.rto_minutes <= 60:
        return "hot-standby"
    elif requirements.rto_minutes <= 240:
        return "warm-standby"
    else:
        return "backup-restore"
```

### Business Impact Assessment

**Calculate Downtime Cost:**
```python
def calculate_downtime_cost(
    service: str,
    downtime_hours: float,
    revenue_per_hour: float,
    customers_affected: int
) -> Dict[str, float]:
    """Calculate financial impact of downtime."""

    # Direct revenue loss
    revenue_loss = downtime_hours * revenue_per_hour

    # Customer churn cost (estimated)
    churn_rate = 0.01 if downtime_hours < 1 else 0.05
    customer_lifetime_value = 10000  # Average CLV
    churn_cost = customers_affected * churn_rate * customer_lifetime_value

    # SLA penalty costs
    sla_penalty = revenue_loss * 0.1  # 10% penalty

    # Reputation damage (estimated)
    reputation_cost = revenue_loss * 0.5

    total_cost = revenue_loss + churn_cost + sla_penalty + reputation_cost

    return {
        "revenue_loss": revenue_loss,
        "churn_cost": churn_cost,
        "sla_penalty": sla_penalty,
        "reputation_cost": reputation_cost,
        "total_cost": total_cost,
        "cost_per_minute": total_cost / (downtime_hours * 60)
    }

# Example: 4-hour outage impact
impact = calculate_downtime_cost(
    service="ai-inference-api",
    downtime_hours=4.0,
    revenue_per_hour=50000,
    customers_affected=1000
)
print(f"Total downtime cost: ${impact['total_cost']:,.2f}")
print(f"Cost per minute: ${impact['cost_per_minute']:,.2f}")
```

### DR Strategy Cost-Benefit Analysis

**Compare DR Strategy Costs:**
```yaml
# DR Strategy Cost Comparison
strategies:
  backup_restore:
    monthly_cost: 500
    rto_hours: 4-8
    rpo_hours: 4-24
    automation_level: low
    suitable_for: [development, staging, non-critical]

  pilot_light:
    monthly_cost: 2000
    rto_hours: 1-2
    rpo_minutes: 15-60
    automation_level: medium
    suitable_for: [business-critical, standard-sla]

  warm_standby:
    monthly_cost: 5000
    rto_minutes: 15-60
    rpo_minutes: 5-15
    automation_level: high
    suitable_for: [mission-critical, high-sla]

  active_active:
    monthly_cost: 10000
    rto_seconds: 0-60
    rpo_seconds: 0-60
    automation_level: very-high
    suitable_for: [zero-downtime, financial, healthcare]
```

---

## Backup Solutions (Advanced Patterns)

### Velero Advanced Configuration

**Multi-Region Backup with Hooks:**
```yaml
# Velero backup with pre/post hooks
apiVersion: velero.io/v1
kind: Backup
metadata:
  name: genops-comprehensive-backup
  namespace: velero
spec:
  # Include specific namespaces
  includedNamespaces:
  - genops
  - genops-production

  # Include specific resources
  includedResources:
  - pods
  - deployments
  - services
  - persistentvolumeclaims
  - configmaps
  - secrets

  # Label selector
  labelSelector:
    matchLabels:
      backup: enabled

  # Storage location
  storageLocation: aws-primary

  # Volume snapshot locations
  volumeSnapshotLocations:
  - aws-us-east-1

  # TTL (30 days)
  ttl: 720h0m0s

  # Hooks for consistent backups
  hooks:
    resources:
    - name: database-backup
      includedNamespaces:
      - genops
      labelSelector:
        matchLabels:
          app: postgresql
      pre:
      - exec:
          command:
          - /bin/bash
          - -c
          - |
            pg_dump -U postgres genops > /backup/genops-$(date +%Y%m%d-%H%M%S).sql
          container: postgresql
          onError: Fail
          timeout: 5m
      post:
      - exec:
          command:
          - /bin/bash
          - -c
          - |
            rm -f /backup/*.sql
          container: postgresql

---
# Schedule backups
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: genops-daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    includedNamespaces:
    - genops
    ttl: 720h0m0s
    storageLocation: aws-primary

---
# Multi-region replication
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: aws-primary
  namespace: velero
spec:
  provider: aws
  objectStorage:
    bucket: genops-velero-backups-us-east-1
    prefix: production
  config:
    region: us-east-1

---
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: aws-dr
  namespace: velero
spec:
  provider: aws
  objectStorage:
    bucket: genops-velero-backups-us-west-2
    prefix: production
  config:
    region: us-west-2
```

### Application-Consistent Backups

**Database Backup with Consistency:**
```yaml
# StatefulSet with backup sidecar
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: genops
spec:
  serviceName: postgresql
  replicas: 3
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
      annotations:
        backup.velero.io/backup-volumes: data
    spec:
      containers:
      - name: postgresql
        image: postgres:15
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        - name: backup
          mountPath: /backup
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secret
              key: password

      # Backup sidecar
      - name: backup-agent
        image: postgres:15
        command:
        - /bin/bash
        - -c
        - |
          while true; do
            timestamp=$(date +%Y%m%d-%H%M%S)
            pg_dump -h localhost -U postgres genops | \
              gzip > /backup/genops-${timestamp}.sql.gz

            # Upload to S3
            aws s3 cp /backup/genops-${timestamp}.sql.gz \
              s3://genops-db-backups/postgresql/

            # Cleanup old local backups
            find /backup -name "*.sql.gz" -mtime +1 -delete

            sleep 3600  # Hourly backups
          done
        volumeMounts:
        - name: backup
          mountPath: /backup
        env:
        - name: AWS_REGION
          value: us-east-1

      volumes:
      - name: backup
        emptyDir: {}

  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Backup Verification and Testing

**Automated Backup Validation:**
```python
#!/usr/bin/env python3
"""Automated backup verification script."""
import subprocess
import json
from datetime import datetime, timedelta

def verify_backup_health():
    """Verify all backups are recent and valid."""
    # Get recent backups
    result = subprocess.run(
        ["velero", "backup", "get", "-o", "json"],
        capture_output=True,
        text=True
    )

    backups = json.loads(result.stdout)

    issues = []

    for backup in backups.get("items", []):
        name = backup["metadata"]["name"]
        status = backup["status"]["phase"]
        completion_time = backup["status"].get("completionTimestamp")

        # Check backup status
        if status != "Completed":
            issues.append(f"Backup {name} failed with status: {status}")
            continue

        # Check backup age
        if completion_time:
            backup_time = datetime.fromisoformat(completion_time.replace("Z", "+00:00"))
            age_hours = (datetime.now(backup_time.tzinfo) - backup_time).total_seconds() / 3600

            if age_hours > 48:
                issues.append(f"Backup {name} is {age_hours:.1f} hours old (stale)")

        # Verify backup contents
        result = subprocess.run(
            ["velero", "backup", "describe", name, "-o", "json"],
            capture_output=True,
            text=True
        )

        details = json.loads(result.stdout)

        if details["status"].get("errors", 0) > 0:
            issues.append(f"Backup {name} has {details['status']['errors']} errors")

    if issues:
        print("❌ Backup verification failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("✅ All backups verified successfully")
    return True

if __name__ == "__main__":
    verify_backup_health()
```

---

## High Availability Architecture (Production-Grade)

### Multi-Zone Deployment with Topology Spread

**Advanced Topology Spread Constraints:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai-ha
  namespace: genops
spec:
  replicas: 9  # 3 per zone
  selector:
    matchLabels:
      app: genops-ai
  template:
    metadata:
      labels:
        app: genops-ai
    spec:
      # Spread evenly across zones
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: genops-ai

      # Spread across nodes within zone
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: genops-ai

      # Anti-affinity for host-level failures
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - genops-ai
            topologyKey: kubernetes.io/hostname

      containers:
      - name: genops-ai
        image: genops-ai:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"

        # Comprehensive health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        startupProbe:
          httpGet:
            path: /startup
            port: 8080
          initialDelaySeconds: 0
          periodSeconds: 10
          failureThreshold: 30

---
# PodDisruptionBudget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: genops-ai-pdb
  namespace: genops
spec:
  minAvailable: 6  # Always maintain 6 of 9 pods
  selector:
    matchLabels:
      app: genops-ai
```

### Cross-Region Active-Active Architecture

**Multi-Region Deployment with Global Load Balancing:**
```yaml
# Primary region (us-east-1)
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-primary
  namespace: genops
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    external-dns.alpha.kubernetes.io/hostname: api-us-east.genops.example.com
    external-dns.alpha.kubernetes.io/ttl: "60"
spec:
  type: LoadBalancer
  selector:
    app: genops-ai
    region: us-east-1
  ports:
  - port: 443
    targetPort: 8080

---
# Secondary region (us-west-2)
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-secondary
  namespace: genops
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    external-dns.alpha.kubernetes.io/hostname: api-us-west.genops.example.com
    external-dns.alpha.kubernetes.io/ttl: "60"
spec:
  type: LoadBalancer
  selector:
    app: genops-ai
    region: us-west-2
  ports:
  - port: 443
    targetPort: 8080

---
# Route53 health check configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: route53-health-config
  namespace: genops
data:
  health-check.yaml: |
    primary:
      endpoint: https://api-us-east.genops.example.com/health
      type: HTTPS
      port: 443
      path: /health
      interval: 30
      failure_threshold: 3

    secondary:
      endpoint: https://api-us-west.genops.example.com/health
      type: HTTPS
      port: 443
      path: /health
      interval: 30
      failure_threshold: 3

    routing_policy:
      type: geolocation_with_failover
      primary_region: us-east-1
      secondary_region: us-west-2
      health_check_enabled: true
```

### Database Replication Strategies

**PostgreSQL Streaming Replication:**
```yaml
# Primary PostgreSQL instance
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql-primary
  namespace: genops
spec:
  serviceName: postgresql-primary
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
      role: primary
  template:
    metadata:
      labels:
        app: postgresql
        role: primary
    spec:
      containers:
      - name: postgresql
        image: postgres:15
        env:
        - name: POSTGRES_REPLICATION_MODE
          value: master
        - name: POSTGRES_REPLICATION_USER
          value: replicator
        - name: POSTGRES_REPLICATION_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-replication
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        - name: config
          mountPath: /etc/postgresql
      volumes:
      - name: config
        configMap:
          name: postgresql-primary-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
# Replica PostgreSQL instance
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql-replica
  namespace: genops
spec:
  serviceName: postgresql-replica
  replicas: 2  # 2 read replicas
  selector:
    matchLabels:
      app: postgresql
      role: replica
  template:
    metadata:
      labels:
        app: postgresql
        role: replica
    spec:
      containers:
      - name: postgresql
        image: postgres:15
        env:
        - name: POSTGRES_REPLICATION_MODE
          value: slave
        - name: POSTGRES_MASTER_SERVICE
          value: postgresql-primary
        - name: POSTGRES_REPLICATION_USER
          value: replicator
        - name: POSTGRES_REPLICATION_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-replication
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
# PostgreSQL configuration for replication
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgresql-primary-config
  namespace: genops
data:
  postgresql.conf: |
    wal_level = replica
    max_wal_senders = 10
    max_replication_slots = 10
    hot_standby = on

  pg_hba.conf: |
    # Allow replication connections
    host replication replicator 0.0.0.0/0 md5
```

---

## Failover Automation (Zero-Touch Recovery)

### DNS-Based Failover with Route53

**Automated Failover Script:**
```python
#!/usr/bin/env python3
"""Automated DNS failover for multi-region deployment."""
import boto3
import requests
from time import sleep

def check_endpoint_health(endpoint: str) -> bool:
    """Check if endpoint is healthy."""
    try:
        response = requests.get(f"{endpoint}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def failover_to_secondary():
    """Failover DNS to secondary region."""
    route53 = boto3.client('route53')

    # Get hosted zone
    hosted_zone_id = "Z1234567890ABC"

    # Update DNS record to point to secondary
    response = route53.change_resource_record_sets(
        HostedZoneId=hosted_zone_id,
        ChangeBatch={
            'Changes': [{
                'Action': 'UPSERT',
                'ResourceRecordSet': {
                    'Name': 'api.genops.example.com',
                    'Type': 'A',
                    'SetIdentifier': 'Secondary',
                    'Failover': 'SECONDARY',
                    'AliasTarget': {
                        'HostedZoneId': 'Z1234567890DEF',
                        'DNSName': 'api-us-west.genops.example.com',
                        'EvaluateTargetHealth': True
                    }
                }
            }]
        }
    )

    print(f"✅ Failover initiated: {response['ChangeInfo']['Id']}")
    return response['ChangeInfo']['Id']

def monitor_and_failover():
    """Continuously monitor and failover if needed."""
    primary_endpoint = "https://api-us-east.genops.example.com"
    secondary_endpoint = "https://api-us-west.genops.example.com"

    failure_count = 0

    while True:
        primary_healthy = check_endpoint_health(primary_endpoint)

        if not primary_healthy:
            failure_count += 1
            print(f"⚠️  Primary endpoint unhealthy (failure {failure_count}/3)")

            if failure_count >= 3:
                print("❌ Primary failed 3 consecutive checks - initiating failover")

                # Verify secondary is healthy before failover
                if check_endpoint_health(secondary_endpoint):
                    failover_to_secondary()
                    break
                else:
                    print("❌ Secondary also unhealthy - manual intervention required")
                    break
        else:
            failure_count = 0
            print("✅ Primary endpoint healthy")

        sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    monitor_and_failover()
```

### Application-Level Failover

**Circuit Breaker with Automatic Region Switching:**
```python
from circuitbreaker import circuit
import requests

class MultiRegionClient:
    """Client with automatic failover between regions."""

    def __init__(self):
        self.primary_url = "https://api-us-east.genops.example.com"
        self.secondary_url = "https://api-us-west.genops.example.com"
        self.current_url = self.primary_url

    @circuit(failure_threshold=5, recovery_timeout=60)
    def call_primary(self, endpoint: str, **kwargs):
        """Call primary region with circuit breaker."""
        response = requests.post(
            f"{self.primary_url}{endpoint}",
            timeout=10,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    def call_with_failover(self, endpoint: str, **kwargs):
        """Call with automatic failover to secondary."""
        try:
            return self.call_primary(endpoint, **kwargs)
        except Exception as e:
            print(f"Primary failed: {e}, failing over to secondary")

            # Failover to secondary
            response = requests.post(
                f"{self.secondary_url}{endpoint}",
                timeout=10,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
```

---

## Data Replication (Multi-Region)

### Persistent Volume Replication with Longhorn

**Longhorn Cross-Region Replication:**
```yaml
# Install Longhorn
apiVersion: v1
kind: Namespace
metadata:
  name: longhorn-system

---
# Longhorn backup target (S3)
apiVersion: v1
kind: Secret
metadata:
  name: longhorn-backup-target
  namespace: longhorn-system
stringData:
  AWS_ACCESS_KEY_ID: "AKIAIOSFODNN7EXAMPLE"
  AWS_SECRET_ACCESS_KEY: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
  AWS_ENDPOINTS: "https://s3.amazonaws.com"

---
# Configure backup target
apiVersion: longhorn.io/v1beta1
kind: Setting
metadata:
  name: backup-target
  namespace: longhorn-system
value: "s3://genops-longhorn-backup@us-east-1/"

---
# Recurring backup for volumes
apiVersion: longhorn.io/v1beta1
kind: RecurringJob
metadata:
  name: backup-genops-volumes
  namespace: longhorn-system
spec:
  cron: "0 */6 * * *"  # Every 6 hours
  task: backup
  groups:
  - default
  retain: 14  # Keep 14 backups
  concurrency: 2
  labels:
    backup-policy: standard
```

### S3 Cross-Region Replication

**Automated S3 Replication Configuration:**
```python
import boto3

def setup_s3_replication(
    source_bucket: str,
    dest_bucket: str,
    source_region: str,
    dest_region: str
):
    """Configure S3 cross-region replication."""
    s3 = boto3.client('s3', region_name=source_region)

    # Enable versioning on both buckets
    s3.put_bucket_versioning(
        Bucket=source_bucket,
        VersioningConfiguration={'Status': 'Enabled'}
    )

    s3_dest = boto3.client('s3', region_name=dest_region)
    s3_dest.put_bucket_versioning(
        Bucket=dest_bucket,
        VersioningConfiguration={'Status': 'Enabled'}
    )

    # Create replication configuration
    replication_config = {
        'Role': 'arn:aws:iam::ACCOUNT:role/S3ReplicationRole',
        'Rules': [{
            'ID': 'ReplicateAll',
            'Status': 'Enabled',
            'Priority': 1,
            'Filter': {},
            'Destination': {
                'Bucket': f'arn:aws:s3:::{dest_bucket}',
                'ReplicationTime': {
                    'Status': 'Enabled',
                    'Time': {'Minutes': 15}
                },
                'Metrics': {
                    'Status': 'Enabled',
                    'EventThreshold': {'Minutes': 15}
                }
            }
        }]
    }

    s3.put_bucket_replication(
        Bucket=source_bucket,
        ReplicationConfiguration=replication_config
    )

    print(f"✅ Replication configured: {source_bucket} → {dest_bucket}")
```

---

## Recovery Procedures (Detailed Runbooks)

### Incident Response Framework

**Complete Recovery Runbook:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dr-runbook-complete
  namespace: genops
data:
  RUNBOOK.md: |
    # GenOps AI Disaster Recovery Runbook

    ## Incident Classification

    ### P0 - Complete Service Outage (RTO: 15 min)
    - All endpoints returning 5xx errors
    - Database unreachable
    - Control plane failure

    ### P1 - Partial Outage (RTO: 1 hour)
    - Single region unavailable
    - Database read replicas down
    - Degraded performance (>2x normal latency)

    ### P2 - Non-Critical (RTO: 4 hours)
    - Non-production environment issues
    - Monitoring gaps
    - Backup failures

    ---

    ## Recovery Procedures

    ### Scenario 1: Complete Cluster Failure

    **Detection:**
    ```bash
    # Check cluster health
    kubectl cluster-info
    kubectl get nodes
    kubectl get pods --all-namespaces
    ```

    **Immediate Actions (0-5 minutes):**
    1. Confirm outage scope
       ```bash
       curl -f https://api.genops.example.com/health
       kubectl get nodes --watch
       ```

    2. Activate incident response
       - Post to #incidents Slack channel
       - Page on-call engineer
       - Start incident log

    3. Check secondary region
       ```bash
       curl -f https://api-us-west.genops.example.com/health
       ```

    **Failover (5-10 minutes):**
    1. Update DNS to secondary region
       ```bash
       python3 scripts/failover-dns.py --to-region us-west-2
       ```

    2. Verify traffic routing
       ```bash
       dig api.genops.example.com
       curl -v https://api.genops.example.com/health
       ```

    3. Monitor secondary region metrics
       ```bash
       kubectl top nodes -n genops
       kubectl get hpa -n genops
       ```

    **Recovery (10-60 minutes):**
    1. Investigate primary region failure
       - Check AWS Service Health Dashboard
       - Review CloudWatch logs
       - Analyze Kubernetes events

    2. If cluster is recoverable, restore services
       ```bash
       # Restart critical pods
       kubectl rollout restart deployment/genops-ai -n genops

       # Verify pod health
       kubectl get pods -n genops -w
       ```

    3. If cluster is lost, restore from backup
       ```bash
       # Create new cluster
       eksctl create cluster -f cluster-config.yaml

       # Restore from Velero
       velero restore create --from-backup genops-daily-latest

       # Verify restoration
       kubectl get all -n genops
       ```

    **Validation (Post-Recovery):**
    - [ ] All services returning 200 OK
    - [ ] Database connectivity verified
    - [ ] GenOps telemetry flowing
    - [ ] SLI metrics within normal range
    - [ ] Customer-facing APIs operational

    ---

    ### Scenario 2: Database Failure

    **Detection:**
    ```bash
    # Check database pods
    kubectl get pods -l app=postgresql -n genops

    # Check connections
    kubectl exec -it postgresql-0 -n genops -- \
      psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
    ```

    **Recovery Steps:**
    1. Promote read replica to primary
       ```bash
       kubectl exec -it postgresql-replica-0 -n genops -- \
         pg_ctl promote -D /var/lib/postgresql/data
       ```

    2. Update application connection strings
       ```bash
       kubectl set env deployment/genops-ai \
         -n genops \
         DATABASE_HOST=postgresql-replica-0.postgresql-replica
       ```

    3. Restore failed primary from backup
       ```bash
       velero restore create \
         --from-backup genops-daily-latest \
         --include-resources persistentvolumeclaims \
         --selector app=postgresql
       ```

    ---

    ### Scenario 3: Data Corruption

    **Detection:**
    - Application errors referencing data integrity
    - Database constraint violations
    - Unexpected query results

    **Recovery Steps:**
    1. Identify corruption timeframe
       ```sql
       SELECT * FROM audit_log
       WHERE timestamp > NOW() - INTERVAL '1 hour'
       ORDER BY timestamp DESC;
       ```

    2. Restore from point-in-time backup
       ```bash
       # List available backups
       velero backup get

       # Restore from specific time
       velero restore create \
         --from-backup genops-daily-20260118020000 \
         --namespace-mappings genops:genops-restore
       ```

    3. Validate restored data
       ```sql
       -- Run data integrity checks
       SELECT COUNT(*) FROM critical_table;
       SELECT * FROM critical_table WHERE id = 'known-good-id';
       ```

    4. Switch to restored namespace
       ```bash
       kubectl patch service genops-ai -n genops \
         -p '{"spec":{"selector":{"namespace":"genops-restore"}}}'
       ```
```

---

## Testing and Validation

### Chaos Engineering with LitmusChaos

**Pod Deletion Chaos Experiment:**
```yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: genops-chaos
  namespace: genops
spec:
  appinfo:
    appns: genops
    applabel: "app=genops-ai"
    appkind: deployment

  engineState: active
  chaosServiceAccount: litmus-admin

  experiments:
  - name: pod-delete
    spec:
      components:
        env:
        - name: TOTAL_CHAOS_DURATION
          value: "60"
        - name: CHAOS_INTERVAL
          value: "10"
        - name: FORCE
          value: "false"
        - name: PODS_AFFECTED_PERC
          value: "25"  # Kill 25% of pods

  - name: pod-network-loss
    spec:
      components:
        env:
        - name: TOTAL_CHAOS_DURATION
          value: "60"
        - name: NETWORK_PACKET_LOSS_PERCENTAGE
          value: "50"
        - name: TARGET_PODS
          value: "genops-ai-.*"
```

### Automated DR Drill Script

**Comprehensive DR Testing:**
```python
#!/usr/bin/env python3
"""Automated disaster recovery drill."""
import subprocess
import time
from datetime import datetime

class DRDrill:
    def __init__(self):
        self.start_time = datetime.now()
        self.results = []

    def run_drill(self):
        """Execute complete DR drill."""
        print("🔥 Starting DR Drill")
        print(f"Start time: {self.start_time}")

        # Phase 1: Simulate failure
        print("\n📍 Phase 1: Simulating primary region failure...")
        self.simulate_failure()

        # Phase 2: Detect and alert
        print("\n📍 Phase 2: Detecting failure...")
        detection_time = self.measure_detection_time()
        self.results.append(f"Detection time: {detection_time}s")

        # Phase 3: Failover
        print("\n📍 Phase 3: Executing failover...")
        failover_time = self.measure_failover_time()
        self.results.append(f"Failover time: {failover_time}s")

        # Phase 4: Validate recovery
        print("\n📍 Phase 4: Validating recovery...")
        self.validate_recovery()

        # Phase 5: Restore primary
        print("\n📍 Phase 5: Restoring primary region...")
        self.restore_primary()

        # Generate report
        self.generate_report()

    def simulate_failure(self):
        """Simulate region failure by scaling down."""
        subprocess.run([
            "kubectl", "scale", "deployment/genops-ai",
            "--replicas=0",
            "-n", "genops",
            "--context", "primary-cluster"
        ])
        time.sleep(10)

    def measure_detection_time(self):
        """Measure how long to detect failure."""
        start = time.time()

        while time.time() - start < 300:  # 5 min timeout
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", "genops"],
                capture_output=True,
                text=True
            )

            if "0/3" in result.stdout:
                return time.time() - start

            time.sleep(5)

        return -1  # Detection failed

    def measure_failover_time(self):
        """Measure failover execution time."""
        start = time.time()

        # Trigger failover
        subprocess.run([
            "python3", "scripts/failover-dns.py",
            "--to-region", "us-west-2"
        ])

        # Wait for DNS propagation
        time.sleep(60)

        return time.time() - start

    def validate_recovery(self):
        """Validate service recovery."""
        import requests

        checks = [
            ("API Health", "https://api.genops.example.com/health"),
            ("Database", "https://api.genops.example.com/db-health"),
            ("Telemetry", "https://api.genops.example.com/metrics")
        ]

        for name, url in checks:
            try:
                response = requests.get(url, timeout=10)
                status = "✅ PASS" if response.status_code == 200 else "❌ FAIL"
                self.results.append(f"{name}: {status}")
            except Exception as e:
                self.results.append(f"{name}: ❌ FAIL - {e}")

    def restore_primary(self):
        """Restore primary region."""
        subprocess.run([
            "kubectl", "scale", "deployment/genops-ai",
            "--replicas=3",
            "-n", "genops",
            "--context", "primary-cluster"
        ])

    def generate_report(self):
        """Generate DR drill report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        print("\n" + "="*60)
        print("📊 DR DRILL REPORT")
        print("="*60)
        print(f"Start: {self.start_time}")
        print(f"End: {end_time}")
        print(f"Total Duration: {duration:.1f}s")
        print("\nResults:")
        for result in self.results:
            print(f"  {result}")
        print("="*60)

if __name__ == "__main__":
    drill = DRDrill()
    drill.run_drill()
```

---

## DR Best Practices Checklist

✅ **Planning:**
- [ ] Define RTO/RPO requirements for each service
- [ ] Document DR strategies and procedures
- [ ] Identify single points of failure
- [ ] Calculate DR costs and budget accordingly
- [ ] Obtain stakeholder approval for DR plan

✅ **Backup:**
- [ ] Automated regular backups (daily minimum)
- [ ] Backup verification and restore testing
- [ ] Off-site backup storage (different region)
- [ ] Backup retention policy (30-90 days typical)
- [ ] Encrypted backups at rest

✅ **High Availability:**
- [ ] Multi-zone deployments for critical services
- [ ] Pod Disruption Budgets configured
- [ ] Health checks and auto-recovery enabled
- [ ] Load balancing across availability zones
- [ ] Stateless design where possible

✅ **Monitoring:**
- [ ] Real-time health monitoring
- [ ] Alerting on availability degradation
- [ ] SLO tracking and reporting
- [ ] Incident response procedures documented
- [ ] On-call rotation and escalation paths

✅ **Testing:**
- [ ] Regular DR drills (quarterly minimum)
- [ ] Documented test results and gaps
- [ ] Chaos engineering practices
- [ ] Performance validation post-recovery
- [ ] Continuous improvement process

✅ **Governance:**
- [ ] Cost tracking continues during DR scenarios
- [ ] Compliance requirements maintained
- [ ] Audit logging during recovery
- [ ] Incident documentation and reporting
- [ ] Post-incident review and improvements

---

## DR Testing Schedule

### Monthly:
- Backup restore validation
- Health check verification
- Runbook review and updates

### Quarterly:
- Full DR drill with failover to secondary region
- Performance testing post-recovery
- RTO/RPO validation
- Team training and tabletop exercises

### Annually:
- Comprehensive DR strategy review
- Cost-benefit analysis
- Compliance audit
- Third-party DR assessment

---

## Next Steps

Ready to implement disaster recovery for GenOps AI? Start with:

1. **Define Requirements** - Document RTO/RPO for your services
2. **Choose DR Strategy** - Select based on criticality and budget
3. **Implement Backups** - Deploy Velero or similar solution
4. **Configure HA** - Multi-zone deployment with proper health checks
5. **Document Procedures** - Create detailed runbooks
6. **Test Regularly** - Schedule and execute DR drills
7. **Monitor and Improve** - Track metrics and refine procedures

Return to [Kubernetes Getting Started](kubernetes-getting-started.md) for the complete deployment overview.

---

## Support

- **Documentation:** [GenOps AI Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Community:** [Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
