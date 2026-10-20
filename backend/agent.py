import os
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

SYSTEM_INSTRUCTION = """You are OrderAssist AI, an empathetic, crisp, and action-oriented Principal Customer Support Agent for an Indian e-commerce platform.

CRITICAL OPERATING RULES:
1. Always rely on your registered Python tools (`get_order_status`, `create_support_ticket`, `initiate_refund_request`) to inspect real MongoDB state before answering any order-specific question. Never guess or fabricate order details.
2. When a customer asks about an order (e.g., ORD-101, ORD-102, ORD-103, ORD-104), call `get_order_status(order_id)` first.
3. If the customer reports a problem (delayed shipment, damaged product, wrong item, billing issue) or asks to raise a complaint/ticket, call `create_support_ticket(order_id, issue_category, description)`.
4. If the customer requests a refund or replacement for an order, call `initiate_refund_request(order_id, reason)`. Note that `ORD-102` (DELAYED) and `ORD-103` (DAMAGED_REPORTED) are eligible for instant automated refunds, whereas `ORD-101` (IN_TRANSIT) and `ORD-104` (DELIVERED cleanly) are not eligible for instant automated refunds.
5. Keep responses natural, warm, and concise (2 to 4 sentences) so they sound natural when read aloud by Web Speech Synthesis (TTS). Mention prices in Indian Rupees (₹) and cite specific Ticket IDs (TKT-XXXX) or Refund References (RFND-XXXXXIN) whenever generated.
"""


def _build_conversation_prompt(history: List[Dict[str, str]], user_message: str) -> str:
    """Formats the last 6 turns from Redis session memory together with the current user message."""
    if not history:
        return f"Customer message: {user_message}"

    lines = ["Previous conversation history (from Redis session memory):"]
    for idx, turn in enumerate(history, start=1):
        lines.append(f"Turn {idx} Customer: {turn.get('user', '')}")
        lines.append(f"Turn {idx} OrderAssist AI: {turn.get('agent', '')}")
    lines.append(f"\nCurrent Customer message: {user_message}")
    return "\n".join(lines)


def _deterministic_fallback_agent(user_message: str) -> str:
    """
    Resilient local ReAct fallback if GEMINI_API_KEY is not configured in the local environment,
    ensuring demos never crash.
    """
    msg_lower = user_message.lower()
    match = re.search(r"ord[-\s]?(\d{3})", msg_lower)
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
    """
    Executes a Gemini 2.5 Flash ReAct / Tool-Calling turn using `from google import genai`.
    Automatically invokes Python tools (`get_order_status`, `create_support_ticket`, `initiate_refund_request`)
    and returns `(reply_text, tool_call_traces)`.
    """
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
