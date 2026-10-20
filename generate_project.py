#!/usr/bin/env python3
"""
OrderAssist AI — Agentic Voice & Chat Customer Support System
Self-Contained Project Repository Generator

Run:
    python3 generate_project.py
"""

import argparse
from pathlib import Path

PROJECT_FILES: dict[str, str] = {
    ".gitignore": """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
.venv/
env/
ENV/

# Environment Variables & Secrets
.env
.env.local
.env.*.local
backend/.env
frontend/.env

# Frontend / Node
node_modules/
frontend/node_modules/
frontend/dist/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*

# IDE & OS
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db
.playwright-cli/
""",

    "backend/requirements.txt": """fastapi==0.115.8
uvicorn[standard]==0.34.0
google-genai==1.5.0
pydantic==2.10.6
pymongo==4.11.1
redis==5.2.1
python-dotenv==1.0.1
httpx==0.28.1
""",

    "backend/.env.example": """# Google Gemini API Key (Required for Gemini 2.5 Flash Agent)
GEMINI_API_KEY=your_google_gemini_api_key_here

# MongoDB Connection URI (Automatically falls back to in-memory store if unreachable)
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=orderassist_ai

# Redis Connection URI (Automatically falls back to in-memory TTL dict if unreachable)
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=3600

# Server Configuration
PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
""",

    "backend/Dockerfile": """FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""",

    "backend/schemas.py": """from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class OrderItemSchema(BaseModel):
    sku: str = Field(..., description="Unique product SKU code")
    name: str = Field(..., description="Product display name")
    qty: int = Field(..., ge=1, description="Quantity ordered")
    price_inr: float = Field(..., ge=0, description="Unit price in INR")


class OrderSchema(BaseModel):
    order_id: str = Field(..., description="Unique Order Identifier e.g., ORD-101")
    customer_name: str = Field(..., description="Full name of the customer")
    customer_email: str = Field(..., description="Customer email address")
    customer_city: str = Field(..., description="Delivery destination city in India")
    items: List[OrderItemSchema] = Field(..., description="Ordered line items")
    total_amount_inr: float = Field(..., description="Total invoice amount in INR")
    status: str = Field(
        ...,
        description="Order status: DELIVERED, IN_TRANSIT, DELAYED, DAMAGED_REPORTED, REFUND_INITIATED",
    )
    payment_method: str = Field(..., description="UPI, Credit Card, NetBanking, or COD")
    carrier: str = Field(..., description="Logistics partner name")
    tracking_awb: str = Field(..., description="Air Waybill tracking number")
    ordered_at: str = Field(..., description="ISO timestamp or date string of order placement")
    delivery_eta: str = Field(..., description="Estimated or actual delivery date/status")
    refund_eligible: bool = Field(..., description="Whether order qualifies for automated refund")
    refund_status: Optional[str] = Field(
        default="NONE",
        description="NONE, PROCESSING, APPROVED, or REJECTED",
    )
    refund_reason: Optional[str] = Field(
        default=None,
        description="Reason recorded if a refund was initiated",
    )


class TicketSchema(BaseModel):
    ticket_id: str = Field(..., description="Generated Support Ticket ID (TKT-XXXX)")
    order_id: str = Field(..., description="Associated Order ID (e.g., ORD-102)")
    issue_category: str = Field(
        ...,
        description="Category: DELAYED_DELIVERY, DAMAGED_ITEM, WRONG_ITEM, BILLING_ISSUE, GENERAL_INQUIRY",
    )
    description: str = Field(..., description="Detailed issue summary reported by the customer")
    status: str = Field(default="OPEN", description="Ticket status: OPEN, IN_PROGRESS, RESOLVED")
    priority: str = Field(default="HIGH", description="Priority level: LOW, MEDIUM, HIGH, CRITICAL")
    created_at: str = Field(..., description="ISO-8601 timestamp when ticket was created")
    resolution_notes: Optional[str] = Field(default=None, description="Agent resolution notes")


class ToolCallTrace(BaseModel):
    tool_name: str = Field(..., description="Name of the Python tool invoked by Gemini")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Dict[str, Any] = Field(default_factory=dict, description="Structured output returned by the tool")


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=3, description="Active Redis conversational session ID")
    message: str = Field(..., min_length=1, description="User query from text or Web Speech ASR")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Active Redis conversational session ID")
    reply: str = Field(..., description="Final synthesized response from the Gemini agent")
    tools_used: List[ToolCallTrace] = Field(
        default_factory=list,
        description="List of tool calls executed during the ReAct turn",
    )
    session_turns_count: int = Field(
        default=1,
        description="Number of conversational turns stored in Redis session memory",
    )
    memory_backend: str = Field(
        default="redis",
        description="Active session memory store ('redis' or 'in_memory_fallback')",
    )
""",

    "backend/memory.py": """import os
import json
import time
import logging
from typing import List, Dict, Any, Tuple
import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("orderassist.memory")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_TURNS = 6

# Graceful in-memory fallback dictionary: { session_id: {"expires_at": float, "turns": list} }
_FALLBACK_MEMORY: Dict[str, Dict[str, Any]] = {}
_redis_client: redis.Redis | None = None
_redis_available: bool = False


def get_redis_client() -> Tuple[redis.Redis | None, bool]:
    \"\"\"Attempts to connect to Redis with a short timeout; falls back to in-memory dict if unreachable.\"\"\"
    global _redis_client, _redis_available
    if _redis_client is not None and _redis_available:
        try:
            _redis_client.ping()
            return _redis_client, True
        except Exception:
            _redis_available = False

    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        client.ping()
        _redis_client = client
        _redis_available = True
        return _redis_client, True
    except Exception as exc:
        logger.info("Redis unreachable (%s). Using resilient in-memory TTL store.", exc)
        _redis_available = False
        return None, False


def _session_key(session_id: str) -> str:
    return f"orderassist:session:{session_id}"


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    \"\"\"
    Retrieves the last 6 conversational turns for `session_id`.
    Each turn is a dictionary: {"user": "...", "agent": "...", "timestamp": "..."}.
    \"\"\"
    client, is_redis = get_redis_client()
    key = _session_key(session_id)

    if is_redis and client is not None:
        try:
            raw = client.get(key)
            if not raw:
                return []
            data = json.loads(raw)
            return data[-MAX_TURNS:] if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Redis read error on %s: %s", key, exc)

    # Fallback path with 3600s TTL enforcement
    entry = _FALLBACK_MEMORY.get(key)
    if not entry:
        return []
    if time.time() > entry.get("expires_at", 0):
        _FALLBACK_MEMORY.pop(key, None)
        return []
    turns = entry.get("turns", [])
    return turns[-MAX_TURNS:]


def save_session_turn(session_id: str, user_msg: str, agent_msg: str) -> Tuple[int, str]:
    \"\"\"
    Appends a new (user_msg, agent_msg) turn to session history, trims to the last 6 turns,
    and persists using Redis SETEX with a 3600-second (1-hour) TTL.
    Returns (turn_count, backend_mode).
    \"\"\"
    history = get_session_history(session_id)
    history.append(
        {
            "user": user_msg,
            "agent": agent_msg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    trimmed = history[-MAX_TURNS:]
    key = _session_key(session_id)

    client, is_redis = get_redis_client()
    if is_redis and client is not None:
        try:
            client.setex(key, SESSION_TTL_SECONDS, json.dumps(trimmed))
            return len(trimmed), "redis"
        except Exception as exc:
            logger.warning("Redis SETEX failed on %s: %s. Writing to fallback.", key, exc)

    _FALLBACK_MEMORY[key] = {
        "expires_at": time.time() + SESSION_TTL_SECONDS,
        "turns": trimmed,
    }
    return len(trimmed), "in_memory_fallback"


def clear_session_history(session_id: str) -> bool:
    \"\"\"Deletes a session key from both Redis and in-memory fallback.\"\"\"
    key = _session_key(session_id)
    _FALLBACK_MEMORY.pop(key, None)
    client, is_redis = get_redis_client()
    if is_redis and client is not None:
        try:
            client.delete(key)
        except Exception:
            pass
    return True
""",

    "backend/tools.py": """import os
import random
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("orderassist.tools")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "orderassist_ai")

# Initial 4 Realistic Indian E-Commerce Seed Orders
INITIAL_ORDERS_SEED: List[Dict[str, Any]] = [
    {
        "order_id": "ORD-101",
        "customer_name": "Aarav Sharma",
        "customer_email": "aarav.sharma@example.in",
        "customer_city": "Bengaluru, Karnataka",
        "items": [
            {
                "sku": "SONY-WH1000XM5",
                "name": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
                "qty": 1,
                "price_inr": 26990.0,
            }
        ],
        "total_amount_inr": 26990.0,
        "status": "IN_TRANSIT",
        "payment_method": "UPI (PhonePe)",
        "carrier": "BlueDart Air Express",
        "tracking_awb": "BD-884920194IN",
        "ordered_at": "2025-05-10",
        "delivery_eta": "Expected Tomorrow by 8:00 PM (Arrived at Bengaluru Sort Facility)",
        "refund_eligible": False,
        "refund_status": "NONE",
        "refund_reason": None,
    },
    {
        "order_id": "ORD-102",
        "customer_name": "Priya Nair",
        "customer_email": "priya.nair@example.in",
        "customer_city": "Kochi, Kerala",
        "items": [
            {
                "sku": "FAB-KURTA-IND",
                "name": "Fabindia Handblock Chanderi Silk Kurta Set (Size M)",
                "qty": 1,
                "price_inr": 4290.0,
            },
            {
                "sku": "FOREST-ESS-MIST",
                "name": "Forest Essentials Facial Tonic Mist Panchpushp",
                "qty": 1,
                "price_inr": 1595.0,
            },
        ],
        "total_amount_inr": 5885.0,
        "status": "DELAYED",
        "payment_method": "HDFC Credit Card",
        "carrier": "Delhivery Surface",
        "tracking_awb": "DLV-449102381IN",
        "ordered_at": "2025-05-03",
        "delivery_eta": "Delayed by 4 days due to monsoon highway congestion near Coimbatore Hub",
        "refund_eligible": True,
        "refund_status": "NONE",
        "refund_reason": None,
    },
    {
        "order_id": "ORD-103",
        "customer_name": "Rohan Verma",
        "customer_email": "rohan.verma@example.in",
        "customer_city": "Mumbai, Maharashtra",
        "items": [
            {
                "sku": "BOAT-NIRVANA-ION",
                "name": "boAt Nirvana Ion ANC True Wireless Earbuds",
                "qty": 1,
                "price_inr": 2299.0,
            },
            {
                "sku": "AMBRANE-65W-GAN",
                "name": "Ambrane 65W GaN Fast Charger Adapter",
                "qty": 1,
                "price_inr": 1499.0,
            },
        ],
        "total_amount_inr": 3798.0,
        "status": "DAMAGED_REPORTED",
        "payment_method": "UPI (Google Pay)",
        "carrier": "Ecom Express",
        "tracking_awb": "ECM-992014412IN",
        "ordered_at": "2025-05-06",
        "delivery_eta": "Delivered on 2025-05-09 (Outer box crushed in transit; damaged unit reported)",
        "refund_eligible": True,
        "refund_status": "NONE",
        "refund_reason": None,
    },
    {
        "order_id": "ORD-104",
        "customer_name": "Ananya Iyer",
        "customer_email": "ananya.iyer@example.in",
        "customer_city": "Chennai, Tamil Nadu",
        "items": [
            {
                "sku": "BLUE-TOKAI-500G",
                "name": "Blue Tokai Attikan Estate Arabica Coffee Beans (500g, French Press)",
                "qty": 2,
                "price_inr": 950.0,
            }
        ],
        "total_amount_inr": 1900.0,
        "status": "DELIVERED",
        "payment_method": "UPI (Paytm)",
        "carrier": "Shadowfax Local",
        "tracking_awb": "SFX-331094820IN",
        "ordered_at": "2025-05-08",
        "delivery_eta": "Delivered on 2025-05-10 at 11:25 AM (Signed by Security Desk)",
        "refund_eligible": False,
        "refund_status": "NONE",
        "refund_reason": None,
    },
]

_mongo_client: MongoClient | None = None
_mongo_available: bool = False
_IN_MEMORY_ORDERS: Dict[str, Dict[str, Any]] = {}
_IN_MEMORY_TICKETS: List[Dict[str, Any]] = []

# Keep a runtime log of tool executions during the current agent turn
_CURRENT_TURN_TOOL_LOG: List[Dict[str, Any]] = []


def reset_turn_tool_log() -> None:
    _CURRENT_TURN_TOOL_LOG.clear()


def get_turn_tool_log() -> List[Dict[str, Any]]:
    return list(_CURRENT_TURN_TOOL_LOG)


def get_db():
    \"\"\"Returns MongoDB database object if MongoDB is running, else None (uses resilient in-memory store).\"\"\"
    global _mongo_client, _mongo_available
    if _mongo_client is not None and _mongo_available:
        try:
            _mongo_client.admin.command("ping")
            return _mongo_client[MONGO_DB_NAME]
        except Exception:
            _mongo_available = False

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1200)
        client.admin.command("ping")
        _mongo_client = client
        _mongo_available = True
        return _mongo_client[MONGO_DB_NAME]
    except Exception as exc:
        logger.info("MongoDB unreachable (%s). Using in-memory document collections.", exc)
        _mongo_available = False
        return None


def seed_initial_data() -> None:
    \"\"\"Seeds ORD-101 through ORD-104 into MongoDB (or fallback store) if empty.\"\"\"
    db = get_db()
    if db is not None:
        if db.orders.count_documents({}) == 0:
            db.orders.insert_many([dict(o) for o in INITIAL_ORDERS_SEED])
            logger.info("Seeded 4 realistic Indian e-commerce orders into MongoDB.")
    else:
        if not _IN_MEMORY_ORDERS:
            for order in INITIAL_ORDERS_SEED:
                _IN_MEMORY_ORDERS[order["order_id"]] = dict(order)
            logger.info("Seeded 4 realistic Indian e-commerce orders into fallback store.")


def list_all_orders() -> List[Dict[str, Any]]:
    \"\"\"Returns all orders sorted by order_id.\"\"\"
    seed_initial_data()
    db = get_db()
    if db is not None:
        docs = list(db.orders.find({}, {"_id": 0}).sort("order_id", 1))
        return docs
    return sorted(_IN_MEMORY_ORDERS.values(), key=lambda x: x["order_id"])


def list_all_tickets() -> List[Dict[str, Any]]:
    \"\"\"Returns all support tickets sorted newest first.\"\"\"
    db = get_db()
    if db is not None:
        return list(db.tickets.find({}, {"_id": 0}).sort("created_at", -1))
    return list(reversed(_IN_MEMORY_TICKETS))


def normalize_order_id(raw_id: str) -> str:
    cleaned = raw_id.strip().upper().replace("#", "")
    if cleaned.isdigit():
        return f"ORD-{cleaned}"
    if not cleaned.startswith("ORD-") and cleaned.startswith("ORD"):
        return f"ORD-{cleaned[3:].lstrip('-')}"
    return cleaned


def get_order_status(order_id: str) -> dict:
    \"\"\"Queries MongoDB for order status, delivery ETA, payment method, and line item details.

    Args:
        order_id: The unique order identifier such as 'ORD-101', 'ORD-102', 'ORD-103', or 'ORD-104'.

    Returns:
        dict: A dictionary containing order details, status, delivery_eta, items, and refund eligibility.
    \"\"\"
    seed_initial_data()
    norm_id = normalize_order_id(order_id)
    db = get_db()
    order = None
    if db is not None:
        order = db.orders.find_one({"order_id": norm_id}, {"_id": 0})
    else:
        order = _IN_MEMORY_ORDERS.get(norm_id)

    if not order:
        result = {
            "found": False,
            "order_id": norm_id,
            "error": f"Order '{norm_id}' was not found. Available sample orders are ORD-101, ORD-102, ORD-103, ORD-104.",
        }
    else:
        result = {
            "found": True,
            **order,
        }

    _CURRENT_TURN_TOOL_LOG.append(
        {
            "tool_name": "get_order_status",
            "args": {"order_id": norm_id},
            "result": result,
        }
    )
    return result


def create_support_ticket(order_id: str, issue_category: str, description: str) -> dict:
    \"\"\"Creates a structured support or escalation ticket in MongoDB with a generated TKT-XXXX ID and status OPEN.

    Args:
        order_id: The order ID associated with the issue (e.g., 'ORD-102', 'ORD-103').
        issue_category: One of 'DELAYED_DELIVERY', 'DAMAGED_ITEM', 'WRONG_ITEM', 'BILLING_ISSUE', 'GENERAL_INQUIRY'.
        description: Detailed summary of the customer's issue or complaint.

    Returns:
        dict: The newly created ticket record including ticket_id, status, and timestamp.
    \"\"\"
    seed_initial_data()
    norm_id = normalize_order_id(order_id)
    ticket_id = f"TKT-{random.randint(1000, 9999)}"
    category_clean = issue_category.strip().upper().replace(" ", "_")

    priority = "HIGH" if category_clean in {"DAMAGED_ITEM", "DELAYED_DELIVERY", "WRONG_ITEM"} else "MEDIUM"
    ticket_doc = {
        "ticket_id": ticket_id,
        "order_id": norm_id,
        "issue_category": category_clean,
        "description": description.strip(),
        "status": "OPEN",
        "priority": priority,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolution_notes": "Assigned to Priority Resolution Queue",
    }

    db = get_db()
    if db is not None:
        db.tickets.insert_one(dict(ticket_doc))
    else:
        _IN_MEMORY_TICKETS.append(dict(ticket_doc))

    result = {
        "success": True,
        "message": f"Support ticket {ticket_id} created for order {norm_id}.",
        "ticket": ticket_doc,
    }
    _CURRENT_TURN_TOOL_LOG.append(
        {
            "tool_name": "create_support_ticket",
            "args": {
                "order_id": norm_id,
                "issue_category": category_clean,
                "description": description,
            },
            "result": result,
        }
    )
    return result


def initiate_refund_request(order_id: str, reason: str) -> dict:
    \"\"\"Verifies if an order is eligible for a refund (e.g., DELAYED or DAMAGED_REPORTED) and updates order & ticket state in MongoDB.

    Args:
        order_id: The order ID for which the customer is requesting a refund (e.g., 'ORD-102' or 'ORD-103').
        reason: Customer's stated reason for requesting the refund.

    Returns:
        dict: Eligibility verification result, updated order status, and refund transaction reference.
    \"\"\"
    seed_initial_data()
    norm_id = normalize_order_id(order_id)
    db = get_db()

    if db is not None:
        order = db.orders.find_one({"order_id": norm_id}, {"_id": 0})
    else:
        order = _IN_MEMORY_ORDERS.get(norm_id)

    if not order:
        result = {
            "success": False,
            "order_id": norm_id,
            "error": f"Order {norm_id} does not exist.",
        }
        _CURRENT_TURN_TOOL_LOG.append(
            {
                "tool_name": "initiate_refund_request",
                "args": {"order_id": norm_id, "reason": reason},
                "result": result,
            }
        )
        return result

    eligible_statuses = {"DELAYED", "DAMAGED_REPORTED", "REFUND_INITIATED"}
    is_eligible = bool(order.get("refund_eligible")) or (order.get("status") in eligible_statuses)

    if not is_eligible:
        result = {
            "success": False,
            "eligible": False,
            "order_id": norm_id,
            "current_status": order.get("status"),
            "message": (
                f"Order {norm_id} is currently '{order.get('status')}' and does not qualify for an instant automated refund. "
                "Only DELAYED or DAMAGED_REPORTED orders qualify automatically."
            ),
        }
        _CURRENT_TURN_TOOL_LOG.append(
            {
                "tool_name": "initiate_refund_request",
                "args": {"order_id": norm_id, "reason": reason},
                "result": result,
            }
        )
        return result

    refund_ref = f"RFND-{random.randint(10000, 99999)}IN"
    updated_fields = {
        "status": "REFUND_INITIATED",
        "refund_status": "APPROVED",
        "refund_reason": reason.strip(),
        "delivery_eta": f"Refund of ₹{order['total_amount_inr']:,.0f} initiated via {order['payment_method']} (Ref: {refund_ref})",
    }

    if db is not None:
        db.orders.update_one({"order_id": norm_id}, {"$set": updated_fields})
        db.tickets.update_many(
            {"order_id": norm_id},
            {"$set": {"status": "RESOLVED", "resolution_notes": f"Automated Refund Approved ({refund_ref})"}},
        )
    else:
        _IN_MEMORY_ORDERS[norm_id].update(updated_fields)
        for tkt in _IN_MEMORY_TICKETS:
            if tkt["order_id"] == norm_id:
                tkt["status"] = "RESOLVED"
                tkt["resolution_notes"] = f"Automated Refund Approved ({refund_ref})"

    # Ensure a ticket record also exists for audit trail if none was created yet
    existing_tickets = [t for t in list_all_tickets() if t["order_id"] == norm_id]
    auto_ticket_id = None
    if not existing_tickets:
        auto_ticket_id = f"TKT-{random.randint(1000, 9999)}"
        ticket_doc = {
            "ticket_id": auto_ticket_id,
            "order_id": norm_id,
            "issue_category": "REFUND_REQUEST",
            "description": f"Automated refund initiated: {reason.strip()}",
            "status": "RESOLVED",
            "priority": "HIGH",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "resolution_notes": f"Refund ₹{order['total_amount_inr']:,.0f} processed ({refund_ref})",
        }
        if db is not None:
            db.tickets.insert_one(dict(ticket_doc))
        else:
            _IN_MEMORY_TICKETS.append(dict(ticket_doc))

    result = {
        "success": True,
        "eligible": True,
        "order_id": norm_id,
        "refund_reference": refund_ref,
        "refund_amount_inr": order["total_amount_inr"],
        "payment_method": order["payment_method"],
        "new_order_status": "REFUND_INITIATED",
        "auto_ticket_id": auto_ticket_id,
        "message": (
            f"Refund of ₹{order['total_amount_inr']:,.0f} for order {norm_id} has been approved and initiated "
            f"to {order['payment_method']} (Reference: {refund_ref})."
        ),
    }
    _CURRENT_TURN_TOOL_LOG.append(
        {
            "tool_name": "initiate_refund_request",
            "args": {"order_id": norm_id, "reason": reason},
            "result": result,
        }
    )
    return result
""",

    "backend/agent.py": """import os
import re
import logging
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import (
    get_order_status,
    create_support_ticket,
    initiate_refund_request,
    reset_turn_tool_log,
    get_turn_tool_log,
)

load_dotenv()

logger = logging.getLogger("orderassist.agent")

SYSTEM_INSTRUCTION = \"\"\"You are OrderAssist AI, an empathetic, crisp, and action-oriented Principal Customer Support Agent for an Indian e-commerce platform.

CRITICAL OPERATING RULES:
1. Always rely on your registered Python tools (`get_order_status`, `create_support_ticket`, `initiate_refund_request`) to inspect real MongoDB state before answering any order-specific question. Never guess or fabricate order details.
2. When a customer asks about an order (e.g., ORD-101, ORD-102, ORD-103, ORD-104), call `get_order_status(order_id)` first.
3. If the customer reports a problem (delayed shipment, damaged product, wrong item, billing issue) or asks to raise a complaint/ticket, call `create_support_ticket(order_id, issue_category, description)`.
4. If the customer requests a refund or replacement for an order, call `initiate_refund_request(order_id, reason)`. Note that `ORD-102` (DELAYED) and `ORD-103` (DAMAGED_REPORTED) are eligible for instant automated refunds, whereas `ORD-101` (IN_TRANSIT) and `ORD-104` (DELIVERED cleanly) are not eligible for instant automated refunds.
5. Keep responses natural, warm, and concise (2 to 4 sentences) so they sound natural when read aloud by Web Speech Synthesis (TTS). Mention prices in Indian Rupees (₹) and cite specific Ticket IDs (TKT-XXXX) or Refund References (RFND-XXXXXIN) whenever generated.
\"\"\"


def _build_conversation_prompt(history: List[Dict[str, str]], user_message: str) -> str:
    \"\"\"Formats the last 6 turns from Redis session memory together with the current user message.\"\"\"
    if not history:
        return f"Customer message: {user_message}"

    lines = ["Previous conversation history (from Redis session memory):"]
    for idx, turn in enumerate(history, start=1):
        lines.append(f"Turn {idx} Customer: {turn.get('user', '')}")
        lines.append(f"Turn {idx} OrderAssist AI: {turn.get('agent', '')}")
    lines.append(f"\\nCurrent Customer message: {user_message}")
    return "\\n".join(lines)


def _deterministic_fallback_agent(user_message: str) -> str:
    \"\"\"
    Resilient local ReAct fallback if GEMINI_API_KEY is not configured in the local environment,
    ensuring demos never crash.
    \"\"\"
    msg_lower = user_message.lower()
    match = re.search(r"ord[-\\s]?(\\d{3})", msg_lower)
    order_id = f"ORD-{match.group(1)}" if match else None

    if not order_id:
        return (
            "Hello! I am OrderAssist AI. Please share your Order ID (for example ORD-101, ORD-102, ORD-103, or ORD-104) "
            "and I can check live tracking, raise a support ticket, or process an instant refund."
        )

    status_info = get_order_status(order_id)
    if not status_info.get("found"):
        return status_info.get("error", f"Could not find order {order_id}.")

    wants_refund = any(w in msg_lower for w in ["refund", "money back", "return", "cancel"])
    wants_ticket = any(w in msg_lower for w in ["ticket", "complaint", "issue", "damaged", "delayed", "broken", "late"])

    reply_parts = [
        f"Order {order_id} for {status_info['customer_name']} ({status_info['items'][0]['name']}) is currently {status_info['status']} ({status_info['delivery_eta']})."
    ]

    if wants_ticket and not wants_refund:
        category = "DAMAGED_ITEM" if "damage" in msg_lower or status_info["status"] == "DAMAGED_REPORTED" else "DELAYED_DELIVERY"
        tkt = create_support_ticket(order_id, category, user_message)
        reply_parts.append(f"I have logged priority support ticket {tkt['ticket']['ticket_id']} for your order.")

    if wants_refund:
        ref_res = initiate_refund_request(order_id, user_message)
        reply_parts.append(ref_res["message"])

    return " ".join(reply_parts)


def run_agent_turn(session_history: List[Dict[str, str]], user_message: str) -> Tuple[str, List[Dict[str, Any]]]:
    \"\"\"
    Executes a Gemini 2.5 Flash ReAct / Tool-Calling turn using `from google import genai`.
    Automatically invokes Python tools (`get_order_status`, `create_support_ticket`, `initiate_refund_request`)
    and returns `(reply_text, tool_call_traces)`.
    \"\"\"
    reset_turn_tool_log()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "your_google_gemini_api_key_here" or api_key == "MY_GEMINI_API_KEY":
        logger.warning("GEMINI_API_KEY not set; using deterministic tool-calling fallback.")
        reply = _deterministic_fallback_agent(user_message)
        return reply, get_turn_tool_log()

    try:
        client = genai.Client(api_key=api_key)
        prompt = _build_conversation_prompt(session_history, user_message)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[get_order_status, create_support_ticket, initiate_refund_request],
                temperature=0.2,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

        reply_text = (response.text or "").strip()
        if not reply_text:
            reply_text = "I have completed the requested database action on your order. Please check the updated telemetry panel."
        return reply_text, get_turn_tool_log()
    except Exception as exc:
        logger.error("Gemini API error (%s); executing deterministic tool fallback.", exc)
        reply = _deterministic_fallback_agent(user_message)
        return reply, get_turn_tool_log()
""",

    "backend/main.py": """import os
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
    \"\"\"Startup hook: automatically seeds 4 realistic Indian e-commerce orders into MongoDB if empty.\"\"\"
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
    \"\"\"Returns all orders from MongoDB (`ORD-101` to `ORD-104`).\"\"\"
    return list_all_orders()


@app.get("/tickets", response_model=List[TicketSchema])
async def get_tickets():
    \"\"\"Returns all support/refund tickets from MongoDB sorted newest first.\"\"\"
    return list_all_tickets()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    \"\"\"
    Executes a multi-turn conversational agent step:
    1. Loads last 6 turns from Redis session history.
    2. Invokes Gemini 2.5 Flash with automatic Python tool calling (`get_order_status`, `create_support_ticket`, `initiate_refund_request`).
    3. Saves the turn back to Redis using `SETEX` (3600s TTL).
    \"\"\"
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
    \"\"\"Clears the Redis conversational memory for the given `session_id`.\"\"\"
    clear_session_history(session_id)
    return {
        "status": "cleared",
        "session_id": session_id,
        "message": f"Session memory for '{session_id}' has been purged.",
    }
""",

    "frontend/package.json": """{
  "name": "orderassist-ai-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --port 5173",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "lucide-react": "^0.475.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.18",
    "@types/react-dom": "^18.3.5",
    "@vitejs/plugin-react": "^4.3.4",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.5.2",
    "tailwindcss": "^3.4.17",
    "vite": "^6.1.0"
  }
}
""",

    "frontend/vite.config.js": """import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://localhost:8000',
      '/orders': 'http://localhost:8000',
      '/tickets': 'http://localhost:8000',
      '/session': 'http://localhost:8000',
    },
  },
});
""",

    "frontend/index.html": """<!doctype html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>OrderAssist AI — Agentic Voice & Chat Customer Support System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
  </head>
  <body class="bg-slate-950 text-slate-100 antialiased">
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
""",

    "frontend/src/index.css": """@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  color-scheme: dark;
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.font-mono,
.tabular-nums {
  font-family: 'JetBrains Mono', 'Plus Jakarta Sans', monospace;
  font-variant-numeric: tabular-nums;
}
""",

    "frontend/src/main.jsx": """import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""",

    "frontend/src/App.jsx": """import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Mic,
  MicOff,
  Send,
  Volume2,
  VolumeX,
  RotateCcw,
  Ticket,
  Terminal,
  Database,
  RefreshCw,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const generateSessionId = () =>
  `sess_${Math.random().toString(36).substring(2, 8)}_${Date.now().toString(36).slice(-4)}`;

const QUICK_PROMPTS = [
  { label: 'Track ORD-101', text: 'Where is my Sony headphones order ORD-101 and when will it arrive?' },
  { label: 'Delayed ORD-102 Refund', text: 'My order ORD-102 is delayed by 4 days. Please create a support ticket and initiate an instant refund.' },
  { label: 'Damaged ORD-103 Claim', text: 'Order ORD-103 arrived with a crushed box and damaged earbuds. Raise a damaged item ticket and process my refund.' },
  { label: 'Check ORD-104 Eligibility', text: 'Can I get an instant automated refund on delivered coffee order ORD-104?' },
];

export default function App() {
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [orders, setOrders] = useState([]);
  const [tickets, setTickets] = useState([]);
  const [messages, setMessages] = useState([
    {
      id: 'welcome-1',
      role: 'agent',
      text: 'Namaste! I am OrderAssist AI, connected to live MongoDB order collections and Redis conversational memory. Ask me about orders ORD-101 through ORD-104 by text or voice (en-IN).',
      tools_used: [],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [sessionTurns, setSessionTurns] = useState(0);
  const [memoryBackend, setMemoryBackend] = useState('redis');
  const [errorBanner, setErrorBanner] = useState(null);

  const recognitionRef = useRef(null);
  const chatFeedRef = useRef(null);

  const fetchTelemetry = useCallback(async () => {
    try {
      const [ordersRes, ticketsRes] = await Promise.all([
        fetch(`${API_BASE}/orders`),
        fetch(`${API_BASE}/tickets`),
      ]);
      if (ordersRes.ok) {
        setOrders(await ordersRes.json());
      }
      if (ticketsRes.ok) {
        setTickets(await ticketsRes.json());
      }
      setErrorBanner(null);
    } catch (err) {
      setErrorBanner('Unable to reach backend server. Ensure FastAPI is running on port 8000.');
    }
  }, []);

  useEffect(() => {
    fetchTelemetry();
  }, [fetchTelemetry]);

  useEffect(() => {
    if (chatFeedRef.current) {
      chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
    }
  }, [messages, isSending]);

  const speakText = useCallback(
    (text) => {
      if (!ttsEnabled || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-IN';
        utterance.rate = 1.02;
        window.speechSynthesis.speak(utterance);
      } catch (e) {
        console.warn('TTS synthesis error:', e);
      }
    },
    [ttsEnabled]
  );

  const handleSendMessage = async (overrideText) => {
    const textToSend = (overrideText !== undefined ? overrideText : input).trim();
    if (!textToSend || isSending) return;

    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (overrideText === undefined) setInput('');
    setIsSending(true);
    setErrorBanner(null);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: textToSend }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const agentMsg = {
        id: `agent-${Date.now()}`,
        role: 'agent',
        text: data.reply,
        tools_used: data.tools_used || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, agentMsg]);
      setSessionTurns(data.session_turns_count || 1);
      setMemoryBackend(data.memory_backend || 'redis');
      speakText(data.reply);

      // Immediately refresh live MongoDB Orders & Tickets after agent turn completes
      await fetchTelemetry();
    } catch (err) {
      setErrorBanner(`Failed to process turn: ${err.message}`);
    } finally {
      setIsSending(false);
    }
  };

  const handleResetSession = async () => {
    try {
      await fetch(`${API_BASE}/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
    } catch (e) {
      // Ignore network error on reset
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    const nextId = generateSessionId();
    setSessionId(nextId);
    setSessionTurns(0);
    setMessages([
      {
        id: `welcome-${Date.now()}`,
        role: 'agent',
        text: `Session reset complete. New Redis session ${nextId} initialized (TTL: 3600s). How can I assist you with your orders today?`,
        tools_used: [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  const toggleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setErrorBanner('Web Speech API is not supported in this browser. Please use Chrome or Edge.');
      return;
    }

    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onerror = (event) => {
      setIsListening(false);
      setErrorBanner(`Voice recognition error: ${event.error}`);
    };
    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript;
      if (transcript) {
        setInput(transcript);
        handleSendMessage(transcript);
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  return (
    <div className="min-h-screen lg:h-screen lg:overflow-hidden bg-slate-950 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between gap-4 md:gap-8 px-6 py-3.5 border-b border-slate-800 bg-slate-950 shrink-0">
        <span className="text-base font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0">
          OrderAssist AI
        </span>
        <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-slate-400">
          <span>Gemini 2.5 Flash ReAct Agent</span>
          <span>·</span>
          <span>MongoDB Collections</span>
          <span>·</span>
          <span>Redis 1h TTL Memory ({memoryBackend})</span>
        </nav>
        <button
          onClick={fetchTelemetry}
          className="px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors flex items-center gap-1.5 whitespace-nowrap shrink-0"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Sync Collections
        </button>
      </header>

      {/* Main Split-Screen Workspace */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 min-h-0 overflow-hidden">
        {/* Left Panel: Live Database & System Telemetry */}
        <section className="lg:col-span-6 border-b lg:border-b-0 lg:border-r border-slate-800 p-6 overflow-y-auto space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <Database className="w-4 h-4 text-emerald-400" />
                Live MongoDB Orders Collection
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Automatically re-fetched after every agent turn · Tabular INR telemetry
              </p>
            </div>
            <span className="text-xs font-mono text-slate-400 tabular-nums">
              {orders.length} records
            </span>
          </div>

          <div className="space-y-3">
            {orders.map((order) => (
              <div
                key={order.order_id}
                className="p-4 rounded-lg border border-slate-800 bg-slate-900/50 hover:border-slate-700 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <span className="font-mono font-semibold text-emerald-400">{order.order_id}</span>
                      <span>·</span>
                      <span className="text-slate-200 font-medium">{order.customer_name}</span>
                      <span>·</span>
                      <span>{order.customer_city}</span>
                    </div>
                    <p className="text-sm font-medium text-slate-100 mt-1">
                      {order.items?.map((i) => `${i.name} (×${i.qty})`).join(', ')}
                    </p>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-sm font-semibold text-slate-100 tabular-nums">
                      <span className="font-sans mr-0.5">₹</span>
                      <span className="font-mono">{Number(order.total_amount_inr).toLocaleString('en-IN')}</span>
                    </div>
                    <div className="text-xs font-mono text-slate-400 mt-0.5">
                      {order.status}
                    </div>
                  </div>
                </div>
                <div className="mt-2.5 pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
                  <span>{order.delivery_eta}</span>
                  <span className="font-mono">
                    {order.carrier} · {order.tracking_awb}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Support Tickets Section */}
          <div className="pt-4 border-t border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                <Ticket className="w-4 h-4 text-amber-400" />
                Live Support & Refund Tickets (`tickets` collection)
              </h2>
              <span className="text-xs font-mono text-slate-400 tabular-nums">
                {tickets.length} active
              </span>
            </div>

            {tickets.length === 0 ? (
              <div className="p-6 rounded-lg border border-slate-800/80 bg-slate-900/30 text-center">
                <p className="text-xs text-slate-400">
                  No support tickets logged yet. Ask the agent to report a delayed or damaged order to watch MongoDB update live.
                </p>
              </div>
            ) : (
              <div className="space-y-2.5">
                {tickets.map((tkt) => (
                  <div
                    key={tkt.ticket_id}
                    className="p-3.5 rounded-lg border border-slate-800 bg-slate-900/60 flex items-start justify-between gap-4"
                  >
                    <div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <span className="font-mono font-semibold text-amber-400">{tkt.ticket_id}</span>
                        <span>·</span>
                        <span className="font-mono text-slate-200">{tkt.order_id}</span>
                        <span>·</span>
                        <span>{tkt.issue_category}</span>
                        <span>·</span>
                        <span className="text-emerald-400 font-mono">{tkt.status}</span>
                      </div>
                      <p className="text-xs text-slate-200 mt-1">{tkt.description}</p>
                      {tkt.resolution_notes && (
                        <p className="text-xs text-slate-400 mt-1">{tkt.resolution_notes}</p>
                      )}
                    </div>
                    <span className="text-xs font-mono text-slate-500 shrink-0 tabular-nums">
                      {tkt.priority}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Right Panel: Voice + Chat Agent Interface */}
        <section className="lg:col-span-6 flex flex-col min-h-[520px] lg:min-h-0 lg:h-full overflow-hidden">
          {/* Session Bar */}
          <div className="px-6 py-3 border-b border-slate-800 bg-slate-900/40 flex items-center justify-between gap-4 shrink-0">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>Redis Session ID:</span>
              <code className="font-mono text-emerald-400">{sessionId}</code>
              <span>·</span>
              <span className="font-mono tabular-nums">{sessionTurns}/6 turns cached</span>
            </div>
            <button
              onClick={handleResetSession}
              className="px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-md transition-colors flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Session
            </button>
          </div>

          {/* Quick Prompts */}
          <div className="px-6 py-2.5 border-b border-slate-800/80 bg-slate-950 flex items-center gap-2 overflow-x-auto shrink-0">
            {QUICK_PROMPTS.map((qp) => (
              <button
                key={qp.label}
                onClick={() => handleSendMessage(qp.text)}
                disabled={isSending}
                className="px-2.5 py-1 text-xs font-medium text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-md transition-colors whitespace-nowrap shrink-0"
              >
                {qp.label}
              </button>
            ))}
          </div>

          {errorBanner && (
            <div className="mx-6 mt-3 p-3 rounded-md bg-rose-950/60 border border-rose-800 text-xs text-rose-200 shrink-0">
              {errorBanner}
            </div>
          )}

          {/* Chat Feed */}
          <div ref={chatFeedRef} className="flex-1 p-6 overflow-y-auto space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-lg px-4 py-3 text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-900 border border-slate-800 text-slate-100'
                  }`}
                >
                  <p>{msg.text}</p>

                  {msg.tools_used && msg.tools_used.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-slate-800 space-y-1.5">
                      <div className="text-xs font-mono text-emerald-400 flex items-center gap-1.5">
                        <Terminal className="w-3.5 h-3.5" />
                        ReAct Tool Execution Trace ({msg.tools_used.length})
                      </div>
                      {msg.tools_used.map((t, idx) => (
                        <div
                          key={idx}
                          className="text-xs font-mono text-slate-300 bg-slate-950/80 px-2.5 py-1.5 rounded border border-slate-800/80"
                        >
                          {t.tool_name}({JSON.stringify(t.args)})
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <span className="text-xs text-slate-500 mt-1 font-mono tabular-nums">
                  {msg.role === 'user' ? 'Customer' : 'OrderAssist AI'} · {msg.timestamp}
                </span>
              </div>
            ))}
            {isSending && (
              <div className="text-xs font-mono text-slate-400">
                OrderAssist AI is running ReAct tool-calling loop...
              </div>
            )}
          </div>

          {/* Input Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="p-4 border-t border-slate-800 bg-slate-900/60 flex items-center gap-2.5 shrink-0"
          >
            <button
              type="button"
              onClick={toggleVoiceInput}
              className={`px-3.5 py-2.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 ${
                isListening
                  ? 'bg-rose-600 text-white'
                  : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
              }`}
            >
              {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              {isListening ? 'Listening (en-IN)...' : 'Speak'}
            </button>

            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about ORD-101, report delayed ORD-102, or request a refund..."
              className="flex-1 bg-slate-950 border border-slate-800 focus:border-emerald-500 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
            />

            <button
              type="button"
              onClick={() => {
                if (ttsEnabled && 'speechSynthesis' in window) {
                  window.speechSynthesis.cancel();
                }
                setTtsEnabled((v) => !v);
              }}
              aria-label={ttsEnabled ? 'Mute voice output' : 'Unmute voice output'}
              className="px-3 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-medium flex items-center gap-1.5 whitespace-nowrap shrink-0"
            >
              {ttsEnabled ? <Volume2 className="w-4 h-4 text-emerald-400" /> : <VolumeX className="w-4 h-4 text-slate-500" />}
              {ttsEnabled ? 'TTS On' : 'Muted'}
            </button>

            <button
              type="submit"
              disabled={isSending || !input.trim()}
              className="px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
            >
              <Send className="w-4 h-4" />
              Send
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}
""",

    "README.md": """# OrderAssist AI — Agentic Voice & Chat Customer Support System

An enterprise-grade, multi-turn **Agentic Voice & Chat Customer Support System** engineered with **FastAPI**, **Google GenAI SDK (`gemini-2.5-flash`)**, **MongoDB (`pymongo`)**, **Redis (`SETEX` session memory)**, and a **React 18 + Tailwind CSS** split-screen telemetry dashboard with browser-native **Web Speech API** (`en-IN` ASR + TTS).

---

## 1. System Architecture Diagram

```
+---------------------------------------------------------------------------------------------------+
|                           FRONTEND: React 18 + Vite + Tailwind CSS                                |
|                                                                                                   |
|  +-------------------------------------------+   +---------------------------------------------+  |
|  | LEFT PANEL: Live DB & System Telemetry    |   | RIGHT PANEL: Voice & Chat Agent Interface   |  |
|  | - Live Orders Table (ORD-101 to ORD-104)  |   | - Active Redis Session ID + Reset Trigger   |  |
|  | - Real-Time Support Tickets (TKT-XXXX)    |   | - Web Speech API Mic Input (en-IN ASR)      |  |
|  | - Auto Re-Syncs on Every Agent Turn       |   | - SpeechSynthesis Voice Output (TTS Toggle) |  |
|  +---------------------+---------------------+   +----------------------+----------------------+  |
+------------------------|------------------------------------------------|-------------------------+
                         | GET /orders, GET /tickets                      | POST /chat, DELETE /session
                         v                                                v
+---------------------------------------------------------------------------------------------------+
|                            BACKEND: Python 3.11+ & FastAPI (main.py)                              |
|                                                                                                   |
|  1. Load Session Context (Last 6 Turns)  ------>  +--------------------------------------------+  |
|  4. Persist Turn via SETEX (3600s TTL)   ------>  | Redis Session Store (memory.py)            |  |
|                                                   | Key: orderassist:session:{session_id}      |  |
|                                                   | Graceful In-Memory TTL Fallback            |  |
|                                                   +--------------------------------------------+  |
|                                                                                                   |
|  2. Execute ReAct Tool-Calling Loop      ------>  +--------------------------------------------+  |
|                                                   | Gemini 2.5 Flash Agent (agent.py)          |  |
|                                                   | google-genai SDK | temperature=0.2         |  |
|                                                   +---------------------+----------------------+  |
|                                                                         |                         |
|                                                   3. Automatic Python Function Calling (tools.py) |
|                                                                         v                         |
|  +---------------------------------------------------------------------------------------------+  |
|  | MongoDB Persistence Layer (`orderassist_ai` Database)                                       |  |
|  | - `get_order_status(order_id)`                          -> Queries `orders` collection      |  |
|  | - `create_support_ticket(order_id, category, desc)`     -> Inserts `TKT-XXXX` into `tickets`|  |
|  | - `initiate_refund_request(order_id, reason)`           -> Verifies eligibility & updates   |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. ReAct Tool-Calling Loop (`backend/agent.py` & `backend/tools.py`)

OrderAssist AI uses the official **Google GenAI SDK** (`from google import genai`) and `gemini-2.5-flash` configured with automatic Python function calling:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[get_order_status, create_support_ticket, initiate_refund_request],
        temperature=0.2,
        system_instruction=SYSTEM_INSTRUCTION,
    ),
)
```

During a single `/chat` request, the model autonomously:
1. **Reasons** over the customer's natural-language or voice-transcribed message and the last 6 conversational turns loaded from Redis.
2. **Acts** by invoking one or more strongly typed Python tools:
   - `get_order_status(order_id: str) -> dict`
   - `create_support_ticket(order_id: str, issue_category: str, description: str) -> dict`
   - `initiate_refund_request(order_id: str, reason: str) -> dict`
3. **Observes** the real MongoDB document state returned by the tool functions and synthesizes a concise, voice-friendly response while returning the structured tool execution trace to the frontend UI.

---

## 3. Redis TTL Session Caching Strategy (`backend/memory.py`)

- **Key Namespace**: `orderassist:session:{session_id}`
- **Sliding Window**: Stores the last **6 conversational turns** (`user`, `agent`, `timestamp`) serialized as JSON.
- **1-Hour Expiration (`SETEX`)**: Every turn refreshes the session key TTL to `3600` seconds via `client.setex(key, 3600, json.dumps(trimmed))`.
- **Zero-Crash Demo Resilience**: If a local Redis server is not running, `memory.py` automatically falls back to an in-memory TTL dictionary enforcing the exact same 3600-second expiration window and 6-turn sliding cap.

---

## 4. MongoDB Schema Design (`orders` & `tickets`)

### `orders` Collection (Seeded automatically on FastAPI startup)
| Order ID | Customer | Item | Status | Refund Eligible |
| :--- | :--- | :--- | :--- | :--- |
| `ORD-101` | Aarav Sharma (Bengaluru) | Sony WH-1000XM5 Headphones (₹26,990) | `IN_TRANSIT` | `False` |
| `ORD-102` | Priya Nair (Kochi) | Fabindia Kurta Set + Forest Essentials Mist (₹5,885) | `DELAYED` | `True` |
| `ORD-103` | Rohan Verma (Mumbai) | boAt Nirvana Ion ANC + 65W GaN Charger (₹3,798) | `DAMAGED_REPORTED` | `True` |
| `ORD-104` | Ananya Iyer (Chennai) | Blue Tokai Attikan Arabica Coffee 500g ×2 (₹1,900) | `DELIVERED` | `False` |

### `tickets` Collection
Stores structured escalations (`ticket_id` formatted as `TKT-XXXX`, `order_id`, `issue_category`, `description`, `status`, `priority`, `created_at`, `resolution_notes`).

---

## 5. Local Setup & Quickstart

### Step 1: Generate the Repository
```bash
python3 generate_project.py
```

### Step 2: Start Backend (FastAPI + Gemini 2.5 Flash)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY in backend/.env
uvicorn main:app --reload --port 8000
```

### Step 3: Start Frontend (React 18 + Vite + Tailwind CSS)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in Chrome or Edge to test live voice (`en-IN`) and chat agent workflows.
""",
}


def generate_repository(base_dir: Path) -> None:
    print(f"Generating OrderAssist AI repository in: {base_dir.resolve()}")
    for rel_path, content in PROJECT_FILES.items():
        target_path = base_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content.lstrip("\\n"), encoding="utf-8")
        print(f"  [CREATED] {rel_path} ({len(content.splitlines())} lines)")
    print("\\nRepository generation complete! All 16 files written successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the OrderAssist AI multi-file project.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Target directory where backend/, frontend/, and README.md will be created (default: current directory).",
    )
    args = parser.parse_args()
    generate_repository(Path(args.output_dir))
