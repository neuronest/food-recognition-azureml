deployment_config = {
    "autoscale_enabled": True,
    "cpu_cores": 4,
    "memory_gb": 32,
    "scoring_timeout_ms": 300000,
    "period_seconds": 30,
    "failure_threshold": 10,
    "timeout_seconds": 30,
    "enable_app_insights": True,
}
conda_packages = ["numpy"]
pip_packages = ["azure-storage-blob", "pyyaml"]
