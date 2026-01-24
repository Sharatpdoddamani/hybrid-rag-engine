"""
FastAPI server for persona-based chat assistant
Enables API access, webhooks, and multi-user support
"""
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from chat_infer import chat_once
import asyncio
from typing import Optional, Dict, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import time

app = FastAPI(
    title="Persona Chat Assistant API",
    description="Lightweight persona-based conversational AI API",
    version="1.0.0"
)

# Enable CORS for web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 60,  # 60 requests per minute
    "requests_per_hour": 500,   # 500 requests per hour
    "burst_size": 10            # Allow burst of 10 requests
}

# Rate limit storage: {ip: {minute_window: count, hour_window: count, last_request: timestamp}}
rate_limit_store: Dict[str, Dict] = defaultdict(lambda: {
    "minute_window": {"start": time.time(), "count": 0},
    "hour_window": {"start": time.time(), "count": 0},
    "burst_tokens": RATE_LIMIT_CONFIG["burst_size"],
    "last_request": time.time()
})

# Response cache: {cache_key: (response, timestamp)}
response_cache: Dict[str, tuple] = {}
CACHE_TTL = 3600  # Cache for 1 hour
CACHE_MAX_SIZE = 1000  # Maximum cache entries

def get_client_ip(request: Request) -> str:
    """Extract client IP address"""
    if request.client:
        return request.client.host
    return "unknown"

def check_rate_limit(ip: str) -> Tuple[bool, str]:
    """Check if request is within rate limits"""
    current_time = time.time()
    client_data = rate_limit_store[ip]
    
    # Reset minute window if expired
    if current_time - client_data["minute_window"]["start"] >= 60:
        client_data["minute_window"] = {"start": current_time, "count": 0}
    
    # Reset hour window if expired
    if current_time - client_data["hour_window"]["start"] >= 3600:
        client_data["hour_window"] = {"start": current_time, "count": 0}
    
    # Refill burst tokens (1 token per second)
    time_since_last = current_time - client_data["last_request"]
    client_data["burst_tokens"] = min(
        RATE_LIMIT_CONFIG["burst_size"],
        client_data["burst_tokens"] + int(time_since_last)
    )
    
    # Check limits
    if client_data["minute_window"]["count"] >= RATE_LIMIT_CONFIG["requests_per_minute"]:
        remaining = 60 - (current_time - client_data["minute_window"]["start"])
        return False, f"Rate limit exceeded: {RATE_LIMIT_CONFIG['requests_per_minute']} requests/minute. Try again in {remaining:.0f} seconds."
    
    if client_data["hour_window"]["count"] >= RATE_LIMIT_CONFIG["requests_per_hour"]:
        remaining = 3600 - (current_time - client_data["hour_window"]["start"])
        return False, f"Rate limit exceeded: {RATE_LIMIT_CONFIG['requests_per_hour']} requests/hour. Try again in {remaining/60:.0f} minutes."
    
    if client_data["burst_tokens"] <= 0:
        return False, "Rate limit exceeded: Too many requests in short time. Please wait a moment."
    
    # Update counters
    client_data["minute_window"]["count"] += 1
    client_data["hour_window"]["count"] += 1
    client_data["burst_tokens"] -= 1
    client_data["last_request"] = current_time
    
    return True, "OK"

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for health checks and docs
    if request.url.path in ["/health", "/docs", "/openapi.json", "/"]:
        return await call_next(request)
    
    ip = get_client_ip(request)
    allowed, message = check_rate_limit(ip)
    
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "Rate limit exceeded", "message": message}
        )
    
    response = await call_next(request)
    
    # Add rate limit headers
    client_data = rate_limit_store[ip]
    response.headers["X-RateLimit-Limit-Minute"] = str(RATE_LIMIT_CONFIG["requests_per_minute"])
    response.headers["X-RateLimit-Remaining-Minute"] = str(RATE_LIMIT_CONFIG["requests_per_minute"] - client_data["minute_window"]["count"])
    response.headers["X-RateLimit-Limit-Hour"] = str(RATE_LIMIT_CONFIG["requests_per_hour"])
    response.headers["X-RateLimit-Remaining-Hour"] = str(RATE_LIMIT_CONFIG["requests_per_hour"] - client_data["hour_window"]["count"])
    
    return response

def get_cache_key(message: str, persona: str, max_tokens: int) -> str:
    """Generate cache key for request"""
    # Normalize inputs for cache key
    key_str = f"{persona.lower().strip()}|{message.lower().strip()}|{max_tokens}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached_response(cache_key: str) -> Optional[str]:
    """Get response from cache if valid"""
    if cache_key in response_cache:
        response, timestamp = response_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return response
        else:
            # Cache expired, remove it
            del response_cache[cache_key]
    return None

def cache_response(cache_key: str, response: str):
    """Cache response"""
    # Remove oldest entries if cache is too large
    if len(response_cache) >= CACHE_MAX_SIZE:
        # Remove 10% oldest entries
        sorted_items = sorted(response_cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_items[:CACHE_MAX_SIZE // 10]:
            del response_cache[key]
    
    response_cache[cache_key] = (response, time.time())

class ChatRequest(BaseModel):
    message: str
    persona: Optional[str] = "You are a helpful, creative conversational assistant. Respond in character to create engaging conversations."
    max_tokens: Optional[int] = 150
    conversation_history: Optional[list] = None
    use_cache: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    success: bool
    cached: bool = False

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the persona-based assistant with caching and rate limiting"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Check cache first (only for simple requests without history)
    cache_key = None
    cached_response = None
    
    if request.use_cache and not request.conversation_history:
        cache_key = get_cache_key(request.message, request.persona, request.max_tokens)
        cached_response = get_cached_response(cache_key)
        
        if cached_response:
            return ChatResponse(response=cached_response, success=True, cached=True)
    
    try:
        # Run in thread pool to not block the async event loop
        loop = asyncio.get_event_loop()
        
        # Handle conversation history if provided
        if request.conversation_history:
            response = await loop.run_in_executor(
                None,
                lambda: chat_once(
                    request.persona, 
                    request.message, 
                    request.conversation_history,
                    request.max_tokens,
                    0.8,  # temperature
                    1,    # num_return_sequences
                    10    # max_context_messages
                )
            )
            # For multi-turn, response is a tuple (response, time, tokens)
            if isinstance(response, tuple):
                response = response[0]
        else:
            response = await loop.run_in_executor(
                None, 
                lambda: chat_once(request.persona, request.message, None, request.max_tokens)
            )
            # For single-turn, response might be a tuple or string
            if isinstance(response, tuple):
                response = response[0]
        
        # Cache response if applicable
        if cache_key and request.use_cache and not request.conversation_history:
            cache_response(cache_key, response)
        
        return ChatResponse(response=response, success=True, cached=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "persona-chat-assistant"}

@app.get("/")
async def root():
    """API documentation"""
    return {
        "message": "Persona Chat Assistant API",
        "endpoints": {
            "POST /chat": "Chat with the bot (with caching and rate limiting)",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
            "GET /stats": "API statistics and cache info"
        },
        "rate_limits": {
            "requests_per_minute": RATE_LIMIT_CONFIG["requests_per_minute"],
            "requests_per_hour": RATE_LIMIT_CONFIG["requests_per_hour"],
            "burst_size": RATE_LIMIT_CONFIG["burst_size"]
        },
        "cache": {
            "enabled": True,
            "ttl_seconds": CACHE_TTL,
            "max_entries": CACHE_MAX_SIZE
        }
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    cache_size = len(response_cache)
    active_clients = len(rate_limit_store)
    
    return {
        "cache": {
            "size": cache_size,
            "max_size": CACHE_MAX_SIZE,
            "hit_rate": "N/A"  # Could be calculated with additional tracking
        },
        "rate_limiting": {
            "active_clients": active_clients,
            "config": RATE_LIMIT_CONFIG
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Persona Chat Assistant API Server...")
    print("=" * 50)
    print("📖 API Docs: http://localhost:8000/docs")
    print("🏥 Health: http://localhost:8000/health")
    print("📊 Stats: http://localhost:8000/stats")
    print("\n✨ Features:")
    print("   - API Rate Limiting (60/min, 500/hour)")
    print("   - Response Caching (1 hour TTL)")
    print("   - Multi-turn Conversation Support")
    print("   - Optimized Context Processing")
    print("\n💡 Use this API with web apps, Discord bots, or mobile apps!")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

