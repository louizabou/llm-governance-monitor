import os
import time
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

# Datadog APM
from ddtrace import tracer, patch_all
patch_all()

# Datadog Metrics API
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_resource import MetricResource
from datadog_api_client.v2.model.metric_series import MetricSeries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datadog configuration
configuration = Configuration()
configuration.api_key["apiKeyAuth"] = os.getenv("DD_API_KEY")

# Cost per 1M tokens (Gemini 2.0 Flash pricing estimate)
COST_PER_1M_INPUT_TOKENS = 0.075
COST_PER_1M_OUTPUT_TOKENS = 0.30

# Safety keywords for basic toxicity detection
SAFETY_KEYWORDS = ["hack", "attack", "exploit", "injection", "ignore previous", "jailbreak", "bypass"]

# Health Score history for rolling average
HEALTH_HISTORY = []

app = FastAPI(title="LLM Governance Monitor", version="2.1.0")

model = None

def get_model():
    global model
    if model is None:
        vertexai.init(
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "i-destiny-461017-g2"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        )
        model = GenerativeModel("gemini-2.0-flash-001")
    return model

def send_metrics(metrics_data: list):
    try:
        with ApiClient(configuration) as api_client:
            api = MetricsApi(api_client)
            api.submit_metrics(body=MetricPayload(series=metrics_data))
    except Exception as e:
        logger.error(f"Failed to send metrics: {e}")

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

def check_safety(text: str) -> dict:
    text_lower = text.lower()
    flagged = [kw for kw in SAFETY_KEYWORDS if kw in text_lower]
    return {
        "is_safe": len(flagged) == 0,
        "safety_score": max(0, 1.0 - (len(flagged) * 0.2)),
        "flagged_keywords": flagged
    }

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT_TOKENS
    output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS
    return round(input_cost + output_cost, 6)

def calculate_health_score(latency_ms: float, cost_usd: float, safety_score: float, has_error: bool = False) -> dict:
    """
    Calculate unified Health Score (0-100) combining all metrics.
    
    Weights (based on enterprise priorities):
    - Performance (latency): 40%
    - Cost efficiency: 30%
    - Safety: 20%
    - Reliability: 10%
    """
    global HEALTH_HISTORY
    
    # Normalize each component to 0-100 scale
    # Performance: 0ms = 100, 10000ms+ = 0
    performance_score = max(0, min(100, 100 - (latency_ms / 100)))
    
    # Cost: $0 = 100, $0.01+ = 0
    cost_score = max(0, min(100, 100 - (cost_usd * 10000)))
    
    # Safety: already 0-1, convert to 0-100
    safety_normalized = safety_score * 100
    
    # Reliability: 100 if no error, 0 if error
    reliability_score = 0 if has_error else 100
    
    # Weighted calculation
    current_score = (
        (performance_score * 0.40) +
        (cost_score * 0.30) +
        (safety_normalized * 0.20) +
        (reliability_score * 0.10)
    )
    
    current_score = round(max(0, min(100, current_score)), 1)
    
    # Rolling average (last 10 requests)
    HEALTH_HISTORY.append(current_score)
    if len(HEALTH_HISTORY) > 10:
        HEALTH_HISTORY.pop(0)
    
    rolling_avg = round(sum(HEALTH_HISTORY) / len(HEALTH_HISTORY), 1)
    
    # Determine status
    if rolling_avg >= 80:
        status = "Excellent"
    elif rolling_avg >= 50:
        status = "Warning"
    else:
        status = "Critical"
    
    return {
        "current": current_score,
        "rolling_avg": rolling_avg,
        "status": status,
        "components": {
            "performance": round(performance_score, 1),
            "cost": round(cost_score, 1),
            "safety": round(safety_normalized, 1),
            "reliability": round(reliability_score, 1)
        }
    }

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Governance Monitor</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: 'Inter', sans-serif; }
            .health-gauge {
                width: 140px;
                height: 140px;
                border-radius: 50%;
                background: conic-gradient(
                    var(--gauge-color) calc(var(--gauge-value) * 1%),
                    #e2e8f0 calc(var(--gauge-value) * 1%)
                );
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
            }
            .health-gauge::before {
                content: '';
                width: 100px;
                height: 100px;
                border-radius: 50%;
                background: white;
                position: absolute;
            }
            .health-gauge-value {
                position: relative;
                z-index: 1;
                font-size: 1.75rem;
                font-weight: 700;
            }
        </style>
    </head>
    <body class="bg-slate-50 text-slate-800">
        <div class="min-h-screen">
            <!-- Header -->
            <header class="bg-white border-b border-slate-200 px-6 py-4">
                <div class="max-w-6xl mx-auto flex items-center justify-between">
                    <div>
                        <h1 class="text-2xl font-semibold text-slate-900">LLM Governance Monitor</h1>
                        <p class="text-sm text-slate-500">Closed-Loop AI Governance</p>
                    </div>
                    <a href="https://us5.datadoghq.com" target="_blank" class="bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 transition">
                        Open Datadog
                    </a>
                </div>
            </header>
            
            <main class="max-w-6xl mx-auto p-6">
                <!-- Health Score + Metrics Row -->
                <div class="flex gap-6 mb-6">
                    <!-- Health Score Card -->
                    <div class="bg-white rounded-xl p-6 shadow-sm border border-slate-200 flex flex-col items-center min-w-[200px]">
                        <div class="text-sm font-medium text-slate-500 mb-3">Health Score</div>
                        <div class="health-gauge" id="healthGauge" style="--gauge-value: 100; --gauge-color: #10b981;">
                            <span class="health-gauge-value" id="healthValue">--</span>
                        </div>
                        <div class="mt-3 text-sm font-semibold" id="healthStatus" style="color: #10b981;">Waiting...</div>
                        <div class="text-xs text-slate-400 mt-1">Avg: <span id="healthRolling">--</span></div>
                    </div>
                    
                    <!-- Metrics Cards -->
                    <div class="flex-1 grid grid-cols-5 gap-4">
                        <div class="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                            <div class="text-sm font-medium text-slate-500 mb-1">Requests</div>
                            <div class="text-2xl font-semibold text-slate-900" id="totalRequests">0</div>
                        </div>
                        <div class="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                            <div class="text-sm font-medium text-slate-500 mb-1">Avg Latency</div>
                            <div class="text-2xl font-semibold text-amber-600" id="avgLatency">0ms</div>
                        </div>
                        <div class="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                            <div class="text-sm font-medium text-slate-500 mb-1">Tokens</div>
                            <div class="text-2xl font-semibold text-slate-900" id="totalTokens">0</div>
                        </div>
                        <div class="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                            <div class="text-sm font-medium text-slate-500 mb-1">Est. Cost</div>
                            <div class="text-2xl font-semibold text-slate-900" id="totalCost">$0.00</div>
                        </div>
                        <div class="bg-white rounded-xl p-4 shadow-sm border border-slate-200">
                            <div class="text-sm font-medium text-slate-500 mb-1">Safety Alerts</div>
                            <div class="text-2xl font-semibold text-rose-600" id="safetyAlerts">0</div>
                        </div>
                    </div>
                </div>
                
                <!-- Health Components -->
                <div class="bg-white rounded-xl p-4 shadow-sm border border-slate-200 mb-6">
                    <div class="text-sm font-medium text-slate-700 mb-3">Health Components</div>
                    <div class="grid grid-cols-4 gap-6">
                        <div>
                            <div class="flex justify-between text-xs text-slate-500 mb-1">
                                <span>Performance (40%)</span>
                                <span id="compPerformance">--%</span>
                            </div>
                            <div class="h-2 bg-slate-200 rounded-full overflow-hidden">
                                <div class="h-full bg-blue-500 transition-all duration-300" id="barPerformance" style="width: 0%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="flex justify-between text-xs text-slate-500 mb-1">
                                <span>Cost Efficiency (30%)</span>
                                <span id="compCost">--%</span>
                            </div>
                            <div class="h-2 bg-slate-200 rounded-full overflow-hidden">
                                <div class="h-full bg-emerald-500 transition-all duration-300" id="barCost" style="width: 0%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="flex justify-between text-xs text-slate-500 mb-1">
                                <span>Safety (20%)</span>
                                <span id="compSafety">--%</span>
                            </div>
                            <div class="h-2 bg-slate-200 rounded-full overflow-hidden">
                                <div class="h-full bg-purple-500 transition-all duration-300" id="barSafety" style="width: 0%"></div>
                            </div>
                        </div>
                        <div>
                            <div class="flex justify-between text-xs text-slate-500 mb-1">
                                <span>Reliability (10%)</span>
                                <span id="compReliability">--%</span>
                            </div>
                            <div class="h-2 bg-slate-200 rounded-full overflow-hidden">
                                <div class="h-full bg-amber-500 transition-all duration-300" id="barReliability" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Chat Area -->
                <div class="bg-white rounded-xl shadow-sm border border-slate-200 mb-6">
                    <div class="border-b border-slate-200 px-4 py-3">
                        <span class="text-sm font-medium text-slate-700">Chat Session</span>
                    </div>
                    <div id="chat" class="p-4 h-64 overflow-y-auto"></div>
                </div>
                
                <!-- Input -->
                <div class="flex gap-3">
                    <input id="msg" type="text" class="flex-1 bg-white border border-slate-300 rounded-lg px-4 py-3 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="Type a message..." onkeypress="if(event.key==='Enter')send()">
                    <button onclick="send()" class="bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition">Send</button>
                    <button onclick="runLoadTest()" class="bg-slate-200 text-slate-700 px-4 py-3 rounded-lg font-medium hover:bg-slate-300 transition">Load Test</button>
                </div>
                
                <!-- Footer -->
                <div class="text-center text-slate-400 text-sm mt-8">
                    Powered by Google Vertex AI (Gemini 2.0) and Datadog
                </div>
            </main>
        </div>
        <script>
            let totalRequests = 0, totalLatency = 0, totalTokens = 0, totalCost = 0, safetyAlerts = 0;
            
            function updateHealthGauge(healthData) {
                const gauge = document.getElementById('healthGauge');
                const value = document.getElementById('healthValue');
                const status = document.getElementById('healthStatus');
                const rolling = document.getElementById('healthRolling');
                
                // Determine color based on score
                let color;
                if (healthData.rolling_avg >= 80) {
                    color = '#10b981'; // green
                } else if (healthData.rolling_avg >= 50) {
                    color = '#f59e0b'; // amber
                } else {
                    color = '#ef4444'; // red
                }
                
                gauge.style.setProperty('--gauge-value', healthData.rolling_avg);
                gauge.style.setProperty('--gauge-color', color);
                value.textContent = Math.round(healthData.current);
                status.textContent = healthData.status;
                status.style.color = color;
                rolling.textContent = healthData.rolling_avg;
                
                // Update component bars
                const components = healthData.components;
                
                document.getElementById('compPerformance').textContent = components.performance + '%';
                document.getElementById('barPerformance').style.width = components.performance + '%';
                
                document.getElementById('compCost').textContent = components.cost + '%';
                document.getElementById('barCost').style.width = components.cost + '%';
                
                document.getElementById('compSafety').textContent = components.safety + '%';
                document.getElementById('barSafety').style.width = components.safety + '%';
                
                document.getElementById('compReliability').textContent = components.reliability + '%';
                document.getElementById('barReliability').style.width = components.reliability + '%';
            }
            
            async function send(customMsg = null) {
                const input = document.getElementById('msg');
                const chat = document.getElementById('chat');
                const msg = customMsg || input.value.trim();
                if (!msg) return;
                
                if (!customMsg) {
                    chat.innerHTML += '<div class="flex justify-end mb-3"><span class="bg-indigo-600 text-white px-4 py-2 rounded-lg max-w-md">' + msg + '</span></div>';
                    input.value = '';
                }
                
                const thinkingId = 'thinking-' + Date.now();
                chat.innerHTML += '<div class="mb-3 text-slate-400" id="' + thinkingId + '">Processing...</div>';
                chat.scrollTop = chat.scrollHeight;
                
                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: msg})
                    });
                    const data = await res.json();
                    
                    document.getElementById(thinkingId)?.remove();
                    
                    totalRequests++;
                    totalLatency += data.latency_ms;
                    totalTokens += data.total_tokens;
                    totalCost += data.cost_usd;
                    if (!data.safety.is_safe) safetyAlerts++;
                    
                    document.getElementById('totalRequests').textContent = totalRequests;
                    document.getElementById('avgLatency').textContent = Math.round(totalLatency/totalRequests) + 'ms';
                    document.getElementById('totalTokens').textContent = totalTokens;
                    document.getElementById('totalCost').textContent = '$' + totalCost.toFixed(4);
                    document.getElementById('safetyAlerts').textContent = safetyAlerts;
                    
                    // Update Health Score
                    if (data.health_score) {
                        updateHealthGauge(data.health_score);
                    }
                    
                    const safetyColor = data.safety.is_safe ? 'text-emerald-600' : 'text-rose-600';
                    const safetyLabel = data.safety.is_safe ? 'Safe' : 'Alert';
                    
                    if (!customMsg) {
                        chat.innerHTML += '<div class="flex justify-start mb-3"><div class="max-w-lg"><span class="bg-slate-100 text-slate-800 px-4 py-2 rounded-lg inline-block">' + data.response + '</span><div class="text-xs text-slate-400 mt-1">' + data.latency_ms + 'ms · ' + data.total_tokens + ' tokens · $' + data.cost_usd.toFixed(5) + ' · <span class="' + safetyColor + '">' + safetyLabel + '</span> · Health: ' + data.health_score.current + '</div></div></div>';
                    }
                    
                } catch(e) {
                    document.getElementById(thinkingId)?.remove();
                    chat.innerHTML += '<div class="text-rose-600 mb-3">Error: ' + e + '</div>';
                }
                chat.scrollTop = chat.scrollHeight;
            }
            
            async function runLoadTest() {
                const prompts = [
                    "What is machine learning?",
                    "Explain cloud computing briefly.",
                    "What is Datadog used for?",
                    "How does AI monitoring work?",
                    "What are LLM tokens?"
                ];
                for (let i = 0; i < prompts.length; i++) {
                    await send(prompts[i]);
                    await new Promise(r => setTimeout(r, 500));
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    safety_check = check_safety(request.message)
    has_error = False
    
    with tracer.trace("llm.chat", service="llm-governance-monitor") as span:
        try:
            input_tokens = estimate_tokens(request.message)
            span.set_tag("llm.input_tokens", input_tokens)
            span.set_tag("llm.safety.is_safe", safety_check["is_safe"])
            span.set_tag("llm.safety.score", safety_check["safety_score"])
            
            response = get_model().generate_content(request.message)
            
            latency_ms = (time.time() - start_time) * 1000
            output_tokens = estimate_tokens(response.text)
            total_tokens = input_tokens + output_tokens
            cost_usd = calculate_cost(input_tokens, output_tokens)
            
            # Calculate Health Score
            health_score = calculate_health_score(
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                safety_score=safety_check["safety_score"],
                has_error=has_error
            )
            
            span.set_tag("llm.output_tokens", output_tokens)
            span.set_tag("llm.total_tokens", total_tokens)
            span.set_tag("llm.latency_ms", latency_ms)
            span.set_tag("llm.cost_usd", cost_usd)
            span.set_tag("llm.health.score", health_score["current"])
            span.set_tag("llm.health.status", health_score["status"])
            span.set_tag("llm.model", "gemini-2.0-flash-001")
            
            timestamp = int(time.time())
            metrics = [
                MetricSeries(
                    metric="llm.tokens.total",
                    type=MetricIntakeType.COUNT,
                    points=[MetricPoint(timestamp=timestamp, value=float(total_tokens))],
                    tags=["service:llm-governance-monitor", "model:gemini-2.0-flash"]
                ),
                MetricSeries(
                    metric="llm.cost.usd",
                    type=MetricIntakeType.GAUGE,
                    points=[MetricPoint(timestamp=timestamp, value=cost_usd)],
                    tags=["service:llm-governance-monitor", "model:gemini-2.0-flash"]
                ),
                MetricSeries(
                    metric="llm.latency.ms",
                    type=MetricIntakeType.GAUGE,
                    points=[MetricPoint(timestamp=timestamp, value=latency_ms)],
                    tags=["service:llm-governance-monitor", "model:gemini-2.0-flash"]
                ),
                MetricSeries(
                    metric="llm.safety.score",
                    type=MetricIntakeType.GAUGE,
                    points=[MetricPoint(timestamp=timestamp, value=safety_check["safety_score"])],
                    tags=["service:llm-governance-monitor"]
                ),
                # Health Score metric
                MetricSeries(
                    metric="llm.health.score",
                    type=MetricIntakeType.GAUGE,
                    points=[MetricPoint(timestamp=timestamp, value=health_score["current"])],
                    tags=["service:llm-governance-monitor", f"status:{health_score['status'].lower()}"]
                ),
            ]
            send_metrics(metrics)
            
            return {
                "response": response.text,
                "latency_ms": round(latency_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "safety": safety_check,
                "health_score": health_score,
                "model": "gemini-2.0-flash-001",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            has_error = True
            span.set_tag("error", True)
            span.set_tag("error.message", str(e))
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.1.0"}

@app.get("/metrics")
async def metrics():
    return {
        "service": "llm-governance-monitor",
        "model": "gemini-2.0-flash-001",
        "datadog_site": os.getenv("DD_SITE"),
        "environment": os.getenv("DD_ENV", "production"),
        "health_history_size": len(HEALTH_HISTORY)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)