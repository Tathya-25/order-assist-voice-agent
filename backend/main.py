import os
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from schemas import ChatRequest, ChatResponse, OrderSchema, TicketSchema, ToolCallTrace
from tools import seed_initial_data, list_all_orders, list_all_tickets
from memory import get_session_history, save_session_turn, clear_session_history
from agent import run_agent_turn

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orderassist.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup hook: automatically seeds 4 realistic Indian e-commerce orders into MongoDB if empty."""
    logger.info("Initializing OrderAssist AI Backend & seeding MongoDB orders if empty...")
    seed_initial_data()
    yield
    logger.info("Shutting down OrderAssist AI Backend.")


app = FastAPI(
    title="OrderAssist AI — Agentic Voice & Chat Customer Support API",
    description="FastAPI + Gemini 2.5 Flash ReAct Tool-Calling Agent + MongoDB + Redis Session Memory",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "orderassist-ai-backend", "model": "gemini-2.5-flash"}


@app.get("/orders", response_model=List[OrderSchema])
async def get_orders():
    """Returns all orders from MongoDB (`ORD-101` to `ORD-104`)."""
    return list_all_orders()


@app.get("/tickets", response_model=List[TicketSchema])
async def get_tickets():
    """Returns all support/refund tickets from MongoDB sorted newest first."""
    return list_all_tickets()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Executes a multi-turn conversational agent step:
    1. Loads last 6 turns from Redis session history.
    2. Invokes Gemini 2.5 Flash with automatic Python tool calling (`get_order_status`, `create_support_ticket`, `initiate_refund_request`).
    3. Saves the turn back to Redis using `SETEX` (3600s TTL).
    """
    session_id = payload.session_id.strip()
    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history = get_session_history(session_id)
    reply_text, tool_traces_raw = run_agent_turn(history, user_message)
    turns_count, memory_backend = save_session_turn(session_id, user_message, reply_text)

    traces = [ToolCallTrace(**trace) for trace in tool_traces_raw]
    return ChatResponse(
        session_id=session_id,
        reply=reply_text,
        tools_used=traces,
        session_turns_count=turns_count,
        memory_backend=memory_backend,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clears the Redis conversational memory for the given `session_id`."""
    clear_session_history(session_id)
    return {
        "status": "cleared",
        "session_id": session_id,
        "message": f"Session memory for '{session_id}' has been purged.",
    }
