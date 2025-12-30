# LLM Governance Monitor - Active Edition

Enterprise-grade, self-healing observability solution for LLM applications. It doesn't just monitor — it actively protects.

## Overview

LLM Governance Monitor implements a closed-loop governance architecture for Large Language Model deployments. Unlike traditional passive monitoring systems, this solution actively intervenes to protect enterprise AI infrastructure.

**Live Demo**: https://llm-governance-monitor-852577507346.us-central1.run.app

**Video Demo**: https://youtu.be/mDsey7eRPZQ

## Key Innovation: Active Governance Engine

Traditional monitoring: Detect → Alert → Human Intervenes → Fix

This solution: Detect → Decide → Act → Adapt (Autonomous)

The system automatically:
1. **Detects** threats in real-time (prompt injection, jailbreaking)
2. **Decides** on appropriate response (allow, throttle, block)
3. **Acts** autonomously to neutralize threats before they reach the LLM
4. **Adapts** its security posture based on observed patterns

## Features

- **Active Governance**: Automatic threat blocking in STRICT MODE
- **Self-Healing Architecture**: Automatic escalation from STANDARD to STRICT mode
- **Health Score**: Unified score combining performance, cost, safety, and reliability
- **Real-Time Metrics**: Token usage, latency, cost tracking
- **Safety Detection**: Prompt injection and harmful content detection
- **Cost Optimization**: Blocked requests don't call the API, saving money
- **Datadog Integration**: Dashboard, 5 monitors, event logging, anomaly detection ML

## Governance Modes

| Mode | Description | Behavior |
|------|-------------|----------|
| STANDARD | Normal operation | Monitor and alert |
| STRICT | Maximum security | Block unsafe requests |

## Architecture
```
User Request
      ↓
┌─────────────────────────────┐
│   GOVERNANCE ENGINE         │
│   (Decision Point)          │
│                             │
│   BLOCK ←── or ──→ ALLOW    │
└─────────────────────────────┘
      ↓                ↓
 "Threat            Vertex AI
  Neutralized"      (Gemini 2.0)
      ↓                ↓
      └────────┬───────┘
               ↓
         Datadog
   (Metrics, Events, APM)
```

## Datadog Integration

### Custom Metrics

| Metric | Description |
|--------|-------------|
| `llm.latency.ms` | Response time in milliseconds |
| `llm.tokens.total` | Total tokens (input + output) |
| `llm.cost.usd` | Estimated cost per request |
| `llm.safety.score` | Safety score (1.0 = safe, 0 = unsafe) |
| `llm.health.score` | Unified health score (0-100) |
| `llm.governance.state_id` | Current mode (0=STANDARD, 2=STRICT) |

### Monitors (5 Detection Rules)

| Monitor | Description |
|---------|-------------|
| High Latency Alert | Triggers when response > 5000ms |
| High Latency Anomaly | ML-based anomaly detection |
| Low Safety Score | Detects prompt injection attempts |
| High Token Usage | Prevents cost overruns |
| Critical Health Score | Monitors overall application health |

### Event Management

Automatic event logging for:
- STRICT MODE activation
- Threat blocked (with keywords, cost saved)
- Safety warnings

## Tech Stack

- **Backend**: Python 3.10, FastAPI
- **AI Model**: Google Vertex AI (Gemini 2.0 Flash)
- **Hosting**: Google Cloud Run
- **Monitoring**: Datadog APM, Metrics API, Events API

## Quick Start
```bash
git clone https://github.com/louizabou/llm-governance-monitor.git
cd llm-governance-monitor

pip install -r requirements.txt

export GOOGLE_CLOUD_PROJECT=your-project-id
export DD_API_KEY=your-datadog-api-key
export DD_SITE=us5.datadoghq.com

uvicorn app:app --reload --port 8000
```

## Project Structure
```
llm-governance-monitor/
├── app.py                    # Main FastAPI application (v3.0)
├── Dockerfile                # Container configuration
├── ARCHITECTURE.md           # Architecture documentation
├── datadog/                  # Datadog configurations
│   ├── dashboard.json
│   └── monitors/
└── README.md
```

## Hackathon

Built for the AI Partner Catalyst Hackathon 2025 (Datadog Challenge)

## License

MIT License

## Author

Louiza Boujida