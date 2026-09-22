# 07. Commercial Playbook: The Freelancer & Enterprise Stack

## 1. The ₹0 Infrastructure Cost Stack

Before raising venture capital or committing to expensive VPS hosting, the entire commercial API is deployed on a zero-idle-cost stack:

```
                          Client API Request
                                  │
                                  ▼
               [Unified FastAPI Gateway / Rate Limiter]
                                  │
         ┌────────────────────────┼────────────────────────┐
         │ Tier 0: FAQ Cache      │ Tier 1: 2B GGUF        │ Tier 2: 7B Burst
         ▼                        ▼                        ▼
[In-Memory / SQLite]     [Oracle ARM 24GB]        [Modal Serverless T4]
Latency: <10ms           Latency: ~150ms          Latency: ~800ms
Cost: ₹0.000             Cost: ₹0.000             Cost: ~$0.0005/query
Capacity: Unlimited      Capacity: ~50k req/mo    Capacity: ~60k req/mo
```

---

## 2. Multi-Account Gemini Pro Key Rotator

* **Throughput**: 3 developer accounts stacked = **4,500 requests/day = 135,000 requests/month**.
* **Bandwidth**: 45 RPM combined throughput.
* **Failover Protocol**: Instant automatic failover on HTTP 429 status with 65-second exponential backoff per individual key.

---

## 3. Freelancer Pricing & Unit Economics

| Plan | Target Audience | Inclusions | Price / Month | Infra Cost | Gross Margin |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Starter** | Small businesses, clinics, coaching institutes | 10,000 queries/month, policy RAG, FAQ automation, email support | **₹5,000 INR** | **₹0.00** | **100%** |
| **Pro / Growth** | High-traffic e-commerce, D2C brands | 50,000 queries/month, multi-turn escalation, custom analytics | **₹18,000 INR** | ~₹200 (Modal burst) | **98.8%** |
| **Overage** | Pay-as-you-go queries beyond tier limits | Direct API bursts | **₹1.00 / query** | ~₹0.04 | **96%** |

With 5 clients on the Starter Plan:
* **Monthly Revenue**: **₹25,000 INR**
* **Infrastructure Cost**: **₹0.00 INR**
* **Net Profit**: **₹25,000 INR (100% margin)**
