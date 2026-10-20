import React, { useState, useEffect, useRef, useCallback } from 'react';
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
