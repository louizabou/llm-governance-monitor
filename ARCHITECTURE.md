# LLM Governance Monitor - Architecture

## Overview

Active, self-healing LLM observability solution that monitors performance, detects threats, and automatically protects LLM applications using Google Cloud and Datadog.

## Key Innovation: Closed-Loop Governance
```
    Traditional:  Detect → Alert → Human Intervenes → Fix
    
    This Solution: Detect → Decide → Act → Adapt (Autonomous)
```

## System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                           USER                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD RUN                             │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              GOVERNANCE ENGINE                          │   │
│   │                                                         │   │
│   │   STANDARD MODE ──────────── STRICT MODE                │   │
│   │   (Monitor)                  (Block Threats)            │   │
│   │                                                         │   │
│   │   Decision: ALLOW ←─────────→ BLOCK                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                     │                    │                      │
│              ┌──────┴──────┐      ┌──────┴──────┐               │
│              │   ALLOW     │      │   BLOCK     │               │
│              │   ↓         │      │   ↓         │               │
│              │ Vertex AI   │      │ "Threat     │               │
│              │ (Gemini)    │      │ Neutralized"│               │
│              └─────────────┘      └─────────────┘               │
│                                                                 │
│   Endpoints:                                                    │
│   ├── GET  /                    → Chat UI                       │
│   ├── POST /chat                → Process + Govern              │
│   ├── POST /governance/simulate-attack → Trigger STRICT         │
│   ├── POST /governance/reset    → Reset to STANDARD             │
│   ├── GET  /governance/status   → Current state                 │
│   └── GET  /health              → Health check                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATADOG                                 │
│                                                                 │
│   Custom Metrics:              Event Management:                │
│   • llm.latency.ms             • STRICT MODE Activated          │
│   • llm.tokens.total           • Threat Blocked                 │
│   • llm.cost.usd               • Safety Warning                 │
│   • llm.safety.score                                            │
│   • llm.health.score           5 Monitors:                      │
│   • llm.governance.state_id    • High Latency Alert             │
│                                • High Latency Anomaly (ML)      │
│                                • Low Safety Score               │
│                                • High Token Usage               │
│                                • Critical Health Score          │
└─────────────────────────────────────────────────────────────────┘
```

## Governance Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| STANDARD | Default / Reset | Monitor, alert, allow all requests |
| STRICT | 2+ safety violations or manual trigger | Block unsafe requests automatically |

## Processing Pipeline
```
Input Message
     │
     ▼
┌──────────────┐
│ Safety Check │ → Detect harmful keywords
└──────────────┘
     │
     ▼
┌──────────────┐
│ Governance   │ → BLOCK or ALLOW decision
│ Decision     │
└──────────────┘
     │
     ├── BLOCK ──────────────────┐
     │                           │
     ▼                           ▼
┌──────────────┐          ┌──────────────┐
│ Vertex AI    │          │ Return       │
│ (Gemini)     │          │ "Blocked"    │
└──────────────┘          │ Cost: $0     │
     │                    └──────────────┘
     ▼
┌──────────────┐
│ Health Score │ → Calculate unified score
└──────────────┘
     │
     ▼
┌──────────────┐
│ Update State │ → Self-healing logic
└──────────────┘
     │
     ▼
┌──────────────┐
│ Send to DD   │ → Metrics + Events
└──────────────┘
     │
     ▼
Return Response + Metrics
```

## Health Score

Unified score (0-100) combining four components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Performance | 40% | Based on latency |
| Cost | 30% | Based on token cost |
| Safety | 20% | Based on safety score |
| Reliability | 10% | Based on errors |

## Custom Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `llm.latency.ms` | Gauge | Response time |
| `llm.tokens.total` | Count | Token consumption |
| `llm.cost.usd` | Gauge | Cost per request |
| `llm.safety.score` | Gauge | Safety score (0-1) |
| `llm.health.score` | Gauge | Health score (0-100) |
| `llm.governance.state_id` | Gauge | Mode (0=STANDARD, 2=STRICT) |

## Monitors (5 Detection Rules)

| Monitor | Condition | Description |
|---------|-----------|-------------|
| High Latency Alert | avg > 5000ms | Slow response detection |
| High Latency Anomaly | ML-based | Anomaly detection |
| Low Safety Score | avg < 0.5 | Prompt injection detection |
| High Token Usage | sum > 10,000 | Cost control |
| Critical Health Score | avg < 50 | Overall health monitoring |

## Event Management

Automatic events sent to Datadog:

| Event | Alert Type | Trigger |
|-------|------------|---------|
| STRICT MODE Activated | Warning | 2+ safety violations |
| Threat Blocked | Success | Unsafe request in STRICT mode |
| Safety Warning | Error | Unsafe prompt detected |

## Safety Detection

Keywords detected:
- hack, attack, exploit
- injection, jailbreak, bypass
- ignore previous, ignore all, disregard, override

## Tech Stack

| Component | Technology |
|-----------|------------|
| Runtime | Python 3.10 |
| Framework | FastAPI |
| AI | Google Vertex AI (Gemini 2.0 Flash) |
| Hosting | Google Cloud Run |
| Monitoring | Datadog APM, Metrics API, Events API |
| Container | Docker + ddtrace |

## URLs

- **Application**: https://llm-governance-monitor-852577507346.us-central1.run.app
- **GitHub**: https://github.com/louizabou/llm-governance-monitor
- **Datadog**: https://us5.datadoghq.com

## Author

Louiza Boujida - AI Partner Catalyst Hackathon 2025
