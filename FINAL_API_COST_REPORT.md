# Final API Expenditure & Cost Reconciliation Report

> **Billing API Provider:** Meta Model API (`https://api.meta.ai/v1/responses`)  
> **Evaluated Foundation Models:** Meta SAM 3.1 (`sam-3.1`) & Muse Wonder (`muse-wonder-1.0`)  
> **Reconciliation Date:** November 4, 2026  
> **Financial Compliance Status:** **STRICTLY WITHIN AUTHORIZED LIMITS**  

---

## 1. Itemized Accounting & Live Billing Reconciliation

| Billing Item | Model Identifier | Unit Rate | Authorized Budget | Billed Volume | Actual Expenditure (USD) | Variance vs Budget | Status |
|:---|:---|:---|---:|---:|---:|---:|:---:|
| **Stage 1: SAM 3.1 Pilot** | `sam-3.1` | $2.50 / 1k imgs ($0.0025 / img) | $0.0750 USD | 30 live calls | **$0.0750 USD** | $0.0000 | **RECONCILED** |
| **Stage 2: SAM 3.1 Full Benchmark** | `sam-3.1` | $2.50 / 1k imgs ($0.0025 / img) | $3.2300 USD | 1,290 live calls | **$3.2250 USD** | -$0.0050 (Under budget) | **RECONCILED** |
| **Pilot Reusable Cache Hits** | `sam-3.1` | $0.0000 (Local hit) | - | 30 cache hits | **$0.0000 USD** | $0.0000 | **RECONCILED** |
| **Subtotal: SAM 3.1 Benchmark** | `sam-3.1` | $0.0025 / live request | $3.3050 USD | 1,320 total imgs | **$3.3000 USD** | -$0.0050 (Under budget) | **RECONCILED** |
| **Muse Wonder Live Validation** | `muse-wonder-1.0` | $4/M in, $20/M out | $2.0000 USD | 5 live requests (9,563 tokens) | **$0.1405 USD** | -$1.8595 (Under budget) | **RECONCILED** |
| **Total Project Expenditure** | **All Models** | - | **$5.3050 USD** | **1,325 live calls** | **$3.4405 USD** | **-$1.8645 (Under budget)** | **RECONCILED** |

---

## 2. Shared Foundation Inference vs Incremental Local Cost

A critical architectural distinction established during the 14-architecture benchmark is the difference between foundation model inference and downstream postprocessing:

1. **Shared Foundation Base Cost:**
   - A single live Meta SAM 3.1 inference call costs **$0.0025 USD**.
   - Downstream postprocessing architectures (`advanced_sam`, `multiscale_sam`, `facade_context`, `dsp_frangi_sam`, `cv_enhanced_sam`) consume the identical SAM output token grammar.
   - **Incremental API Cost:** **$0.0000 USD**. Postprocessing and filtering operate entirely on local CPU memory.
2. **Hybrid Inspection Cost:**
   - `hybrid` incurs the base SAM API cost ($0.0025 USD) plus local U-Net execution ($0.0000 USD), totaling **$0.0025 USD per inspection**.
3. **Offline Standalone Cost:**
   - `unet_standalone` and `sahi_standalone` run exclusively on local edge compute, incurring **$0.0000 USD per inspection**.
4. **Semantic Verification Cost:**
   - Selective Muse Wonder verification runs only on ambiguous candidates (max 10 candidates per inspection).
   - Mean live token cost per candidate: **$0.0281 USD**.
   - With selective filtering (~1 to 3 ambiguous candidates per inspection), typical Wonder overhead is **$0.028 - $0.084 USD**.

---

## 3. Telemetry & Billing Audit Traceability

1. **Request Traceability:**
   - Every live API transaction has a persistent record under `runs/<run_id>/` storing `request.json`, `response.json`, and `meta.json`.
   - Exact Meta request IDs (`resp_...`) and token usage blocks (`input_tokens`, `output_tokens`, `reasoning_tokens`) are recorded.
2. **Zero Inadvertent Spend:**
   - Automated test suites (`pytest`) utilize `ExplodingTransport` and `FakeTransport`, making it physically impossible for automated regression runs to touch external billing endpoints.
3. **Hard Budget Enforcement:**
   - `WonderVerifier` enforces an internal programmatic spend ceiling (`global_spend_ceiling_usd = 2.00`). If cumulative spend reaches this threshold, subsequent calls immediately return `status="skipped_budget"`.
