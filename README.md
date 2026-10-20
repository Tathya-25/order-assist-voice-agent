# OrderAssist AI — Agentic Voice & Chat Customer Support System

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
