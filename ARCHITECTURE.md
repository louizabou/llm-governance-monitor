# LLM Governance Monitor - Architecture

## Overview

Real-time LLM observability application that monitors chatbot performance, detects safety issues, and tracks costs using Google Cloud and Datadog.

## System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                           USER                                   │
│                      (Web Browser)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD RUN                              │
│                                                                  │
│   FastAPI Application                                            │
│   ├── GET  /        → Chat UI (HTML/JS)                         │
│   ├── POST /chat    → Process message, return metrics           │
│   ├── GET  /health  → Health check                              │
│   └── GET  /metrics → Debug endpoint                            │
│                                                                  │
│   Core Functions:                                                │
│   • Token estimation (input/output)                             │
│   • Cost calculation (USD)                                      │
│   • Safety scoring (prompt injection detection)                 │
│   • Latency measurement                                         │
└─────────────────────────────────────────────────────────────────┘
             │                              │
             ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────────────┐
│   GOOGLE VERTEX AI      │    │          DATADOG                │
│                         │    │                                 │
│   Model: Gemini 2.0     │    │   Custom Metrics:               │
│   Flash                 │    │   • llm.latency.ms              │
│                         │    │   • llm.tokens.total            │
│   Location: us-central1 │    │   • llm.cost.usd                │
│                         │    │   • llm.safety.score            │
│                         │    │                                 │
│                         │    │   Monitors:                     │
│                         │    │   • High Latency (>5s)          │
│                         │    │   • Low Safety (<0.5)           │
│                         │    │   • High Tokens (>10k)          │
└─────────────────────────┘    └─────────────────────────────────┘
```

## Components

### 1. Frontend (HTML/JavaScript)

Single-page chat interface with real-time metrics display.

**Features**:
- Message input and chat history
- Live metrics dashboard (requests, latency, tokens, cost, safety alerts)
- Load test button for demo purposes
- Direct link to Datadog dashboard

**Technologies**: Tailwind CSS, vanilla JavaScript, async/await

### 2. Backend (FastAPI)

Python application handling chat requests and metrics collection.

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves chat UI |
| `/chat` | POST | Processes messages via Gemini, returns response + metrics |
| `/health` | GET | Health check for Cloud Run |
| `/metrics` | GET | Debug endpoint for metrics config |

**Processing Pipeline**:
```
Input Message
     │
     ▼
┌─────────────┐
│ Safety Check │ → Flag harmful keywords
└─────────────┘
     │
     ▼
┌─────────────┐
│ Token Count │ → Estimate input tokens
└─────────────┘
     │
     ▼
┌─────────────┐
│ Gemini API  │ → Generate response
└─────────────┘
     │
     ▼
┌─────────────┐
│ Metrics     │ → Calculate latency, output tokens, cost
└─────────────┘
     │
     ▼
┌─────────────┐
│ Send to DD  │ → Push custom metrics to Datadog
└─────────────┘
     │
     ▼
Return Response + Metrics
```

### 3. AI Model (Vertex AI)

**Configuration**:
- Model: `gemini-2.0-flash-001`
- Location: `us-central1`
- Lazy loading for faster cold starts

**Cost Estimation** (per 1M tokens):
- Input: $0.075
- Output: $0.30

### 4. Observability (Datadog)

**Custom Metrics**:

| Metric | Type | Description |
|--------|------|-------------|
| `llm.latency.ms` | Gauge | Response time |
| `llm.tokens.total` | Count | Token consumption |
| `llm.cost.usd` | Gauge | Cost per request |
| `llm.safety.score` | Gauge | Safety score (0-1) |

**Monitors**:

| Monitor | Threshold | Action |
|---------|-----------|--------|
| High Latency | avg > 5000ms | Email alert |
| Low Safety | avg < 0.5 | Email alert |
| High Tokens | sum > 10000 | Email alert |

**Dashboard Widgets**:
- Total Tokens (Query Value)
- Total Cost USD (Query Value)
- Safety Score (Gauge)
- LLM Latency Over Time (Timeseries)
- Monitor Status (Summary)

## Safety Detection

Basic prompt injection detection using keyword matching.

**Flagged Keywords**:
- hack, attack, exploit
- injection, jailbreak, bypass
- "ignore previous"

**Safety Score Calculation**:
```
score = 1.0 - (flagged_keywords_count * 0.2)
```

## Deployment

**Platform**: Google Cloud Run (serverless)

**Environment Variables**:
```
GOOGLE_CLOUD_PROJECT=project-id
GOOGLE_CLOUD_LOCATION=us-central1
DD_API_KEY=datadog-api-key
DD_SITE=us5.datadoghq.com
DD_SERVICE=llm-governance-monitor
DD_ENV=production
```

**Container**: Python 3.10-slim with ddtrace instrumentation

## Data Flow
```
1. User sends message
2. Frontend POST to /chat
3. Backend checks safety
4. Backend calls Gemini API
5. Backend calculates metrics
6. Backend sends metrics to Datadog
7. Backend returns response + metrics to frontend
8. Frontend updates UI
9. Datadog evaluates monitors
10. If threshold breached → Alert triggered
```

## URLs

- **Application**: https://llm-governance-monitor-852577507346.us-central1.run.app
- **GitHub**: https://github.com/louizabou/llm-governance-monitor
- **Datadog**: https://us5.datadoghq.com

## Author

Louiza Boujida - AI Partner Catalyst Hackathon 2025

## Components

### 1. Frontend (HTML/JavaScript)
- Simple chat interface using Tailwind CSS
- Real-time message display
- Async communication with backend

### 2. Backend (FastAPI)
- **GET /**: Serves the chat UI
- **POST /chat**: Processes messages via Gemini
- **GET /health**: Health check endpoint

### 3. AI Model (Vertex AI)
- Model: `gemini-2.0-flash-001`
- Location: `us-central1`
- Lazy loading for faster cold starts

### 4. Observability (Datadog)
- **APM**: Distributed tracing
- **Monitors**:
  - High Latency Alert (>5s)
  - High Error Rate Alert (>10%)
  - Request Spike Alert (>10 req/s)
- **Incident Management**: Actionable alerts

## Tech Stack

| Component | Technology |
|-----------|------------|
| Runtime | Python 3.10 |
| Framework | FastAPI |
| AI | Google Vertex AI (Gemini 2.0) |
| Hosting | Google Cloud Run |
| Monitoring | Datadog APM |
| Container | Docker |

## Deployment
```bash
gcloud run deploy llm-governance-monitor \
  --project i-destiny-461017-g2 \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## URLs

- **Application**: https://llm-governance-monitor-852577507346.us-central1.run.app
- **GitHub**: https://github.com/louizabou/llm-governance-monitor
- **Datadog Dashboard**: https://us5.datadoghq.com

