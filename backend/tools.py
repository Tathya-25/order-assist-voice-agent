import os
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
    """Returns MongoDB database object if MongoDB is running, else None (uses resilient in-memory store)."""
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
    """Seeds ORD-101 through ORD-104 into MongoDB (or fallback store) if empty."""
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
    """Returns all orders sorted by order_id."""
    seed_initial_data()
    db = get_db()
    if db is not None:
        docs = list(db.orders.find({}, {"_id": 0}).sort("order_id", 1))
        return docs
    return sorted(_IN_MEMORY_ORDERS.values(), key=lambda x: x["order_id"])


def list_all_tickets() -> List[Dict[str, Any]]:
    """Returns all support tickets sorted newest first."""
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
    """Queries MongoDB for order status, delivery ETA, payment method, and line item details.

    Args:
        order_id: The unique order identifier such as 'ORD-101', 'ORD-102', 'ORD-103', or 'ORD-104'.

    Returns:
        dict: A dictionary containing order details, status, delivery_eta, items, and refund eligibility.
    """
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
    """Creates a structured support or escalation ticket in MongoDB with a generated TKT-XXXX ID and status OPEN.

    Args:
        order_id: The order ID associated with the issue (e.g., 'ORD-102', 'ORD-103').
        issue_category: One of 'DELAYED_DELIVERY', 'DAMAGED_ITEM', 'WRONG_ITEM', 'BILLING_ISSUE', 'GENERAL_INQUIRY'.
        description: Detailed summary of the customer's issue or complaint.

    Returns:
        dict: The newly created ticket record including ticket_id, status, and timestamp.
    """
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
    """Verifies if an order is eligible for a refund (e.g., DELAYED or DAMAGED_REPORTED) and updates order & ticket state in MongoDB.

    Args:
        order_id: The order ID for which the customer is requesting a refund (e.g., 'ORD-102' or 'ORD-103').
        reason: Customer's stated reason for requesting the refund.

    Returns:
        dict: Eligibility verification result, updated order status, and refund transaction reference.
    """
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
