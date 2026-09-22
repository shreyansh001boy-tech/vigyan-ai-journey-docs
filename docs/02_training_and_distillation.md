# 02. Distributed Training, FSDP & Distillation Cascades

## 1. Top-Down Distillation Cascade

The knowledge transfer pipeline operates as a unidirectional waterfall:

```
[Vigyan-32B Titan Teacher]
         │
         │  Generates step-by-step synthetic CoT reasoning paths
         │  Ground truth filtering on physics & mathematical derivations
         ▼
[Curated CoT Staging: Vigyan-Defence-STEM-CoT-Master]
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
[Vigyan-7B Scholar SFT]                   [Vigyan-2B Edge SFT]
(ml.g5.2xlarge - 1x A10G)                 (ml.g5.2xlarge - 1x A10G)
Job duration: ~1.5 hours                  Job duration: ~1.0 hour
Cost: ~$1.82 USD                          Cost: ~$1.21 USD
```

---

## 2. AWS SageMaker BYOS (Bring-Your-Own-Script) Architecture

To prevent vendor lock-in and eliminate unnecessary managed platform overhead, all training jobs were executed using SageMaker's native PyTorch Deep Learning Containers (`763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker`):

### Automated Bootstrap Execution Flow:
1. Local launcher packages training source code into a clean `sourcedir.tar.gz`.
2. S3 bucket staging: Uploads code bundle to `s3://sagemaker-us-east-1-117097040493/source/`.
3. SageMaker On-Demand instance provisioning (`ml.g5.2xlarge` with 1x NVIDIA A10G, 24 GB VRAM).
4. Container entrypoint unbundles code, provisions dependencies, streams weights from Hugging Face, executes training, and exports fine-tuned LoRA adapters directly to private HF repositories.
5. Instance self-terminates immediately upon job completion to enforce strict zero idle spend.

---

## 3. PEFT & LoRA Hyperparameter Configurations

* **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
* **Rank ($r$)**: 16 (for 7B) / 8 (for 2B).
* **Alpha ($\alpha$)**: 32.
* **Dropout**: 0.05.
* **Precision**: BFloat16 mixed precision with FlashAttention-2.
* **Batch Size**: Effective batch size of 16 using gradient accumulation steps of 4.
