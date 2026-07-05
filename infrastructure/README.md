# infrastructure

Infrastructure-as-Code for Signalium Quant production deployments.

## Structure

```
infrastructure/
  terraform/     Cloud resource provisioning (AWS/GCP/Azure/RunPod)
  ansible/       Server configuration management
  cloud-init/    VM bootstrap scripts for VPS and RunPod instances
```
