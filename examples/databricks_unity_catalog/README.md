# Databricks Unity Catalog + GenOps

**Get enterprise data governance in 2 minutes** with zero code changes to your existing Databricks applications.

## ⚡ Quick Start (2 minutes)

**Already have Databricks credentials?** Try this immediately:

```bash
pip install genops[databricks]
python quick_demo.py
```

**Don't have credentials yet?** Follow the [5-minute setup guide](../../docs/databricks-unity-catalog-quickstart.md).

## 🎯 What You Get

✅ **Real-time cost tracking** across SQL warehouses, compute clusters, and storage  
✅ **Automatic data lineage** capture for all Unity Catalog operations  
✅ **Team-based governance** with budget controls and policy enforcement  
✅ **Zero code changes** required - works with your existing applications  
✅ **OpenTelemetry telemetry** compatible with 15+ observability platforms

## 📚 Examples & Guides

### ⚡ Immediate Value (2 minutes)

**[quick_demo.py](quick_demo.py)** ⭐ **START HERE**
- Zero configuration required if you have Databricks environment variables set
- Shows immediate governance value with real examples
- Copy-paste ready code that works right now

### 🏃 Getting Started (5 minutes each)

**[setup_validation.py](setup_validation.py)**
- Validate your Databricks setup with detailed diagnostics
- Actionable error messages with fix suggestions
- Run this first before trying other examples

**[basic_tracking.py](basic_tracking.py)**  
- Learn core governance tracking with Unity Catalog
- Team cost attribution and data lineage examples
- Foundation patterns for real applications

**[auto_instrumentation.py](auto_instrumentation.py)**
- Zero-code setup that works with existing applications  
- Automatic governance without changing your code
- Perfect for existing Databricks users

### 📚 Complete Documentation

- **[5-Minute Quickstart](../../docs/databricks-unity-catalog-quickstart.md)** - Get started fast
- **[Complete Integration Guide](../../docs/integrations/databricks-unity-catalog.md)** - Everything you need to know

## 🏃 Running Examples

**Quick path for first-time users:**

```bash
# 1. Quick demo (2 minutes) - immediate value  
python quick_demo.py

# 2. Validate setup (30 seconds) - check configuration
python setup_validation.py  

# 3. Learn basics (5 minutes) - foundation patterns
python basic_tracking.py

# 4. Zero-code setup (5 minutes) - existing applications  
python auto_instrumentation.py
```

**Run all examples:**
```bash
./run_all_examples.sh
```

## 🔍 Troubleshooting

**❌ "DATABRICKS_HOST not set"**
```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your_personal_access_token"
```

**❌ "Unity Catalog not accessible"**  
- Ensure Unity Catalog is enabled in your workspace
- Verify your user has Unity Catalog permissions

**❌ Still having issues?**
- 🚀 Try the [5-Minute Quickstart](../../docs/databricks-unity-catalog-quickstart.md)
- 📧 [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues) with error details
- 💬 [Community Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**⚡ Get started in 2 minutes:** `python quick_demo.py`