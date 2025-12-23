# LLM Governance Monitor

Enterprise-grade LLM observability with real-time monitoring, safety detection, and cost tracking.

## Overview

LLM Governance Monitor provides production-ready observability for Large Language Model deployments. It monitors LLM performance, tracks costs, and detects prompt injection attempts in real-time.

**Live Demo**: https://llm-governance-monitor-852577507346.us-central1.run.app

## Features

- **Real-Time Metrics**: Token usage, latency, and cost tracking displayed live
- **Safety Detection**: Prompt injection and harmful content detection
- **Cost Monitoring**: Estimated cost per request in USD
- **Smart Alerting**: 3 detection rules for latency, safety, and token spikes
- **Datadog Dashboard**: Comprehensive observability dashboard

## Architecture
```
User → Cloud Run (FastAPI) → Vertex AI (Gemini 2.0)
              ↓
         Datadog
    (Metrics + Monitors)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Datadog Integration

### Custom Metrics

| Metric | Description |
|--------|-------------|
| `llm.latency.ms` | Response time in milliseconds |
| `llm.tokens.total` | Total tokens (input + output) |
| `llm.cost.usd` | Estimated cost per request |
| `llm.safety.score` | Safety score (1.0 = safe, 0 = unsafe) |

### Monitors

| Monitor | Condition |
|---------|-----------|
| High Latency Alert | > 5000ms |
| Low Safety Score | < 0.5 |
| High Token Usage | > 10,000 tokens |

## Tech Stack

- **Backend**: Python 3.10, FastAPI
- **AI Model**: Google Vertex AI (Gemini 2.0 Flash)
- **Hosting**: Google Cloud Run
- **Monitoring**: Datadog APM + Custom Metrics

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

llm-governance-monitor/
├── app.py                    # Main FastAPI application
├── Dockerfile                # Container configuration
├── ARCHITECTURE.md           # Architecture documentation
├── datadog/                  # Datadog configurations
│   ├── dashboard.json
│   ├── monitor-high-latency.json
│   ├── monitor-low-safety.json
│   └── monitor-high-tokens.json
└── traffic_generator.py      # Load testing script


## Hackathon

Built for the AI Partner Catalyst Hackathon (Google Cloud + Datadog).

## License

MIT License

## Author

Louiza Boujida