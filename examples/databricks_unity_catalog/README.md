# Databricks Unity Catalog Examples

This directory contains comprehensive examples demonstrating GenOps governance telemetry integration with Databricks Unity Catalog for enterprise data governance, cost intelligence, and compliance automation.

## 🏗️ What is Databricks Unity Catalog?

**Databricks Unity Catalog is an enterprise data governance platform** that provides unified governance, security, and metadata management across all data assets in your Databricks workspace. Think of it as a comprehensive control plane for all your data platforms.

### Why Use Unity Catalog + GenOps?

- **🏛️ Unified Data Governance**: Track data lineage, access patterns, and compliance across all catalogs with cost attribution
- **💰 Multi-Workspace Cost Intelligence**: Understand compute, storage, and query costs across all workspaces and teams
- **👥 Team-Based Governance**: Enforce budget controls and policy compliance with team attribution
- **📊 Compliance Automation**: Automated PII detection, retention policies, and audit trail generation
- **🚀 Enterprise Scale**: Handle large-scale data operations with real-time governance and cost tracking
- **🏛️ Cross-Stack Integration**: Unified governance across SQL warehouses, compute clusters, and ML workloads

**Perfect for**: Data engineering teams, data scientists, platform engineers, and enterprises managing large-scale data operations.

## 🚀 Quick Start

### Step 1: Prerequisites & Setup (5 minutes)

**New to GenOps + Unity Catalog?** Start here for a complete setup:

1. **Install GenOps with Databricks support:**
   ```bash
   pip install genops[databricks]
   ```

2. **Get your Databricks credentials:**
   - Get your workspace URL (e.g., `https://your-workspace.cloud.databricks.com`)
   - Create a personal access token at Settings → User Settings → Access Tokens

3. **Configure environment variables:**
   ```bash
   # Required: Databricks workspace and authentication
   export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
   export DATABRICKS_TOKEN="your_personal_access_token"
   
   # Recommended for full governance
   export GENOPS_TEAM="your-team"
   export GENOPS_PROJECT="your-project"
   export GENOPS_ENVIRONMENT="dev"  # dev, staging, prod
   ```

### Step 2: Validate Your Setup (30 seconds) ⭐ **START HERE**

**Run this FIRST** to ensure everything is working:

```bash
python setup_validation.py
```

✅ **Expected result:** `Overall Status: PASSED`  
❌ **If you see errors:** Check the [Troubleshooting section](#-troubleshooting) below

### Step 3: Choose Your Learning Path

**✨ New to data governance?** → [5-Minute Quickstart Guide](../../docs/databricks-unity-catalog-quickstart.md)  
**🏃 Want to try examples?** → Continue with [Level 1 examples](#level-1-getting-started-5-minutes-each) below  
**📚 Need complete documentation?** → [Comprehensive Integration Guide](../../docs/integrations/databricks-unity-catalog.md)

## 📚 Examples by Complexity

### Level 1: Getting Started (5 minutes each)

**🎯 Goal:** Understand basics of Unity Catalog + GenOps integration  
**👤 Perfect for:** First-time users, developers new to data governance

**1. [setup_validation.py](setup_validation.py)** ⭐ **Run this first**
- ✅ Verify your Databricks + GenOps setup across dependencies and configuration
- ✅ Validate workspace connectivity, Unity Catalog access, and governance features  
- ✅ Get immediate feedback on configuration issues with actionable fixes
- ✅ Test data governance features and cost tracking accuracy
- **Next:** Try `basic_tracking.py` to see governance in action

**2. [basic_tracking.py](basic_tracking.py)** 
- 🏛️ Simple data operations with Unity Catalog governance tracking
- 💰 Introduction to cost attribution and team-based tracking
- 📊 Basic metadata operations with governance attributes
- 🚀 Minimal code changes for maximum governance capability
- **Next:** Try `auto_instrumentation.py` for zero-code setup

**3. [auto_instrumentation.py](auto_instrumentation.py)**
- 🤖 Zero-code setup using GenOps auto-instrumentation with Unity Catalog
- 📈 Automatic cost tracking for existing Databricks applications
- 🔄 Drop-in governance integration with no code changes required
- **Next:** Ready for Level 2 - Data Governance

### Level 2: Data Governance & Cost Intelligence (30 minutes each)

**🎯 Goal:** Build expertise in data governance and multi-workspace cost optimization  
**👤 Perfect for:** Data engineers, platform engineers ready for advanced workflows  
**📋 Prerequisites:** Complete Level 1 examples

**4. [advanced_features.py](advanced_features.py)**
- 🔄 Complete data lineage tracking across catalogs, schemas, and tables
- 📊 Advanced metadata operations with compliance automation
- 🎛️ Policy enforcement and governance rule validation
- 📈 Cross-workspace cost analysis and attribution
- **Next:** Try `cost_optimization.py` to optimize spending

**5. [cost_optimization.py](cost_optimization.py)**
- 💰 Multi-workspace cost analysis and optimization strategies
- 🚨 Budget monitoring and alerts across teams and projects
- 📊 Resource efficiency analysis for SQL warehouses and compute clusters
- 🔮 Cost forecasting based on historical data patterns
- **Next:** Ready for Level 3 - Enterprise Deployment

### Level 3: Enterprise Governance & Compliance (2 hours each)

**🎯 Goal:** Master enterprise-grade features and production deployment patterns  
**👤 Perfect for:** Data platform teams, enterprise deployments, compliance officers  
**📋 Prerequisites:** Complete Level 2 examples

**6. [compliance_automation.py](compliance_automation.py)**
- 🏛️ Enterprise compliance automation with Unity Catalog governance
- 🔍 Automated PII detection, data classification, and retention policies
- 👥 Multi-team governance with role-based access controls
- 📈 Advanced audit trail generation and compliance reporting
- **Next:** Try `production_patterns.py` for enterprise deployment

**7. [production_patterns.py](production_patterns.py)**
- 🏭 Enterprise-ready Unity Catalog deployment patterns with governance
- ⚡ High-availability data operations with multi-workspace failover
- 🔧 Context managers for complex data workflows with cost tracking
- 🛡️ Policy enforcement and governance automation for production operations
- 🚀 CI/CD integration patterns for data governance workflows
- **Next:** Deploy in production with [Enterprise Deployment Guide](../../docs/enterprise/databricks-unity-catalog-enterprise-deployment.md)

## 🎯 Use Case Examples

Each example includes:
- ✅ **Complete working code** you can run immediately
- ✅ **Data governance demonstrations** with real Unity Catalog scenarios
- ✅ **Cost optimization strategies** for compute, storage, and data transfer
- ✅ **Multi-workspace patterns** showcasing enterprise governance
- ✅ **Error handling** and graceful degradation for production use
- ✅ **Performance considerations** for large-scale data operations
- ✅ **Comments explaining** GenOps + Unity Catalog integration points

## 🏃 Running Examples

### 🎯 Recommended Path for First-Time Users

**Follow this exact sequence for the best learning experience:**

```bash
# Step 1: Validate setup (REQUIRED)
python setup_validation.py      # ⭐ Always run this first!

# Step 2: Choose your path
# For beginners → Start with basic tracking
python basic_tracking.py        # Learn the fundamentals

# For existing Unity Catalog users → Try auto-instrumentation  
python auto_instrumentation.py  # Zero-code governance integration

# Step 3: Build expertise (after completing Level 1)
python advanced_features.py     # Complete data governance lifecycle
python cost_optimization.py     # Cost-aware planning

# Step 4: Advanced usage (after completing Level 2)
python compliance_automation.py # Enterprise compliance features
python production_patterns.py   # Enterprise deployment patterns
```

### ⚡ Quick Options

**New to everything?**
```bash
# Complete beginner path (30 minutes total)
python setup_validation.py && python basic_tracking.py && python auto_instrumentation.py
```

**Already know Unity Catalog?**
```bash  
# Advanced user path (2 hours total)
python setup_validation.py && python auto_instrumentation.py && python production_patterns.py
```

**Want to try everything?**
```bash
# Run all examples with comprehensive validation
./run_all_examples.sh
```

## 📊 What You'll Learn & Success Checkpoints

### ✅ **Level 1 Success Criteria (Getting Started)**
After completing Level 1, you should be able to:
- [ ] Run `python setup_validation.py` and see `Overall Status: PASSED` 
- [ ] Track a basic data operation with cost attribution
- [ ] See governance metadata in your Unity Catalog dashboard
- [ ] Understand automatic cost tracking for your data operations

**🎯 Success Validation:**
```bash
# You should see cost and governance data in output
python basic_tracking.py | grep -E "(Cost|Team|Governance)"
```

### ✅ **Level 2 Success Criteria (Data Governance)**  
After completing Level 2, you should be able to:
- [ ] Manage complete data governance lifecycles with lineage tracking
- [ ] Set up multi-workspace cost monitoring and budget alerts
- [ ] Run compliance checks with automated policy enforcement
- [ ] Generate cost optimization recommendations across workspaces

**🎯 Success Validation:**
```bash
# Should show data governance and cost optimization completed
python advanced_features.py && python cost_optimization.py
```

### ✅ **Level 3 Success Criteria (Enterprise Governance)**
After completing Level 3, you should be able to:
- [ ] Deploy enterprise governance patterns with automated compliance
- [ ] Configure multi-workspace governance with role-based access
- [ ] Implement automated PII detection and data classification
- [ ] Integrate with CI/CD pipelines and enterprise monitoring systems

**🎯 Success Validation:**  
```bash
# Should complete without errors and show enterprise metrics
python production_patterns.py | tail -20
```

### 📚 **Knowledge Areas Covered**

**Data Governance Excellence:**
- How to track data lineage across catalogs, schemas, and tables with cost intelligence
- Cost optimization strategies for SQL warehouses and compute clusters  
- Multi-workspace governance with team-based attribution
- Compliance automation and policy enforcement for enterprise data operations

**GenOps Governance Excellence:**
- Cross-workspace cost attribution and team tracking
- Unified telemetry across your entire data platform
- Policy enforcement and compliance automation
- Enterprise-ready governance patterns for data workflows

**Production Data Platform Deployment Patterns:**
- High-availability data operations with multi-workspace failover
- Auto-scaling data workloads with cost awareness
- Performance optimization and resource efficiency analysis
- Integration with existing data platforms and observability systems

## 🔍 Troubleshooting

### Common Issues

### 🆘 **Most Common Issues (90% of problems)**

**❌ "Databricks authentication failed" or connection errors**
```bash
# Step 1: Verify your workspace URL and token
echo "Workspace: $DATABRICKS_HOST"
echo "Token: ${DATABRICKS_TOKEN:0:10}..."  # Show first 10 chars

# Step 2: Test basic connectivity
curl -H "Authorization: Bearer $DATABRICKS_TOKEN" \
     "$DATABRICKS_HOST/api/2.0/clusters/list"

# Step 3: Check Unity Catalog access
python -c "
from databricks.sdk import WorkspaceClient
client = WorkspaceClient()
print('Catalogs:', len(list(client.catalogs.list())))
"
```

**❌ "databricks-sdk module not found" or import errors**
```bash
# Step 1: Install with correct extras
pip install genops[databricks]

# Step 2: Verify installation
python -c "import databricks.sdk, genops; print('✅ Ready to go!')"

# Step 3: If still failing, try upgrading
pip install --upgrade genops[databricks] databricks-sdk
```

**❌ "GenOps validation failed" - setup issues**
```bash
# Step 1: Run detailed validation to see specific errors
python setup_validation.py --detailed --connectivity --governance

# Step 2: Enable debug logging for more info
export GENOPS_LOG_LEVEL=DEBUG
python setup_validation.py

# Step 3: Check prerequisites one by one
python -c "
import os
print('Host:', '✅ Set' if os.getenv('DATABRICKS_HOST') else '❌ Missing')
print('Token:', '✅ Set' if os.getenv('DATABRICKS_TOKEN') else '❌ Missing')
"
```

### 🔧 **Less Common Issues**

**❌ Cost tracking not working:**
```bash
# Enable detailed logging and retry
export GENOPS_LOG_LEVEL=DEBUG
python basic_tracking.py
```

**❌ Examples running but no governance data:**
```bash
# Check your team/project settings
echo "Team: $GENOPS_TEAM, Project: $GENOPS_PROJECT"
# If empty, set them:
export GENOPS_TEAM="your-team"
export GENOPS_PROJECT="your-project"
```

**❌ Unity Catalog permissions issues:**
```bash
# Test Unity Catalog access
python -c "
from databricks.sdk import WorkspaceClient
client = WorkspaceClient()
try:
    catalogs = list(client.catalogs.list())
    print(f'✅ Unity Catalog access: {len(catalogs)} catalogs')
except Exception as e:
    print(f'❌ Unity Catalog error: {e}')
"
```

### 🆘 **Still Having Issues?**

**📧 Get Help:**
- 📚 **First:** Check [Complete Integration Guide](../../docs/integrations/databricks-unity-catalog.md) for detailed solutions
- 🚀 **Alternative:** Try [5-Minute Unity Catalog Quickstart](../../docs/databricks-unity-catalog-quickstart.md) for simpler approach
- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues) with full error details
- 💬 **Community:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions) for questions

**📋 When Asking for Help, Include:**
1. Output from `python setup_validation.py --detailed`
2. Your Python version: `python --version`  
3. Your operating system and version
4. Complete error messages (copy-paste, don't screenshot)
5. What you were trying to do when the error occurred

## 🌟 Next Steps

### ✅ After Completing Level 1 (Beginner)
- **Integrate patterns** from `basic_tracking.py` into your existing data operations
- **Add auto-instrumentation** to existing Unity Catalog applications for instant governance
- **Read:** [5-Minute Unity Catalog Quickstart Guide](../../docs/databricks-unity-catalog-quickstart.md) for additional examples

### ✅ After Completing Level 2 (Intermediate)  
- **Implement cost optimization** strategies from `cost_optimization.py` in your team
- **Set up data governance lifecycle management** for better data operations
- **Configure compliance automation** and team-based governance controls

### ✅ After Completing Level 3 (Advanced)
- **Deploy enterprise patterns** using `production_patterns.py` as a template
- **Read:** [Enterprise Deployment Guide](../../docs/enterprise/databricks-unity-catalog-enterprise-deployment.md)
- **Consider:** [Migration from other data governance platforms](../../docs/migration-guides/databricks-from-competitors.md)

### 📚 Continue Learning
- **Comprehensive Guide**: [Complete Unity Catalog Integration Documentation](../../docs/integrations/databricks-unity-catalog.md)
- **Other Integrations**: Explore [Bedrock](../bedrock/), [LangChain](../langchain/), and [WandB](../wandb/) examples
- **Community**: Join discussions at [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

## 🎯 Decision Guide: Is Unity Catalog + GenOps Right for You?

### ✅ **Perfect for Unity Catalog + GenOps:**
- **Data Engineering Teams** wanting comprehensive governance with cost intelligence
- **Platform Engineers** who need to optimize data infrastructure costs
- **Data Scientists** requiring governance controls for ML data workflows
- **Enterprises** needing compliance automation and multi-workspace governance
- **Organizations** wanting cost attribution and budget controls for data operations

### 🤔 **Consider alternatives:**
- **Simple data workflows** with minimal governance needs → Try [OpenAI](../openai/) or [Anthropic](../anthropic/) examples
- **Basic data operations** without compliance requirements → Standard Databricks might be sufficient
- **Non-data use cases** → Explore other [GenOps integrations](../../docs/integrations/)

### 💡 **Still unsure?**
- **Start with:** [5-Minute Unity Catalog Quickstart](../../docs/databricks-unity-catalog-quickstart.md) to see if it fits your needs
- **Compare:** Check our [migration guide](../../docs/migration-guides/databricks-from-competitors.md) if you're using Apache Atlas, Azure Purview, or AWS Glue
- **Ask questions:** Join our [community discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**Ready to get started?** Run `python setup_validation.py` to validate your setup and begin your GenOps + Unity Catalog journey!