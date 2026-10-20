from typing import List, Optional, Dict, Any
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
