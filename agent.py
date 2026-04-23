import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from tools import mock_lead_capture

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # Keeps the demo runnable before dependencies are installed.
    END = None
    StateGraph = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PLATFORMS = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "facebook": "Facebook",
    "linkedin": "LinkedIn",
    "twitch": "Twitch",
    "x": "X",
    "twitter": "X",
}


class AgentState(TypedDict, total=False):
    history: List[Dict[str, str]]
    intent: str
    lead: Dict[str, Optional[str]]
    pending_field: Optional[str]
    lead_captured: bool
    current_message: str
    retrieved_docs: List[Dict[str, Any]]
    response: str


def _load_knowledge_base() -> Dict[str, Any]:
    kb_path = Path(__file__).with_name("knowledge_base.json")
    with kb_path.open("r", encoding="utf-8") as f:
        return json.load(f)


KB = _load_knowledge_base()
if load_dotenv:
    load_dotenv()


def _empty_state() -> AgentState:
    return {
        "history": [],
        "intent": "inquiry",
        "lead": {"name": None, "email": None, "platform": None},
        "pending_field": None,
        "lead_captured": False,
    }


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _knowledge_documents() -> List[Dict[str, str]]:
    if "documents" in KB:
        return KB["documents"]

    return [
        {
            "id": "pricing",
            "title": "Pricing and features",
            "content": KB.get("pricing_text", ""),
        },
        {"id": "refunds", "title": "Refund policy", "content": KB.get("refund_policy", "")},
        {"id": "support", "title": "Support policy", "content": KB.get("support_policy", "")},
    ]


def retrieve_documents(query: str, limit: int = 2) -> List[Dict[str, Any]]:
    """Small local RAG retriever using keyword overlap against the JSON KB."""
    query_tokens = _tokenize(query)
    boosts = {
        "price": {"pricing", "plan", "cost", "monthly", "basic", "pro"},
        "pricing": {"pricing", "plan", "cost", "monthly", "basic", "pro"},
        "cost": {"pricing", "price", "plan", "monthly", "basic", "pro"},
        "plan": {"pricing", "price", "cost", "monthly", "basic", "pro"},
        "basic": {"pricing", "price", "plan", "monthly"},
        "pro": {"pricing", "price", "plan", "monthly", "4k", "captions"},
        "refund": {"refund", "policy", "days"},
        "support": {"support", "24", "7", "pro"},
        "feature": {"feature", "video", "caption", "resolution", "4k", "720p"},
    }
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        expanded_tokens.update(boosts.get(token, set()))

    ranked = []
    for doc in _knowledge_documents():
        haystack = f"{doc['title']} {doc['content']}"
        doc_tokens = _tokenize(haystack)
        score = len(expanded_tokens & doc_tokens)
        ranked.append({**doc, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    matches = [doc for doc in ranked if doc["score"] > 0]
    return matches[:limit] if matches else ranked[:1]


def _llm_intent(message: str, active_state: Optional[AgentState] = None) -> Optional[str]:
    if (
        OpenAI is None
        or not os.getenv("OPENAI_API_KEY")
        or os.getenv("USE_LLM_INTENT", "").lower() != "true"
    ):
        return None

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's latest AutoStream message as exactly one "
                        "label: greeting, inquiry, high_intent, or lead_collection. "
                        "Use lead_collection when the agent is already collecting a lead."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": message,
                            "pending_field": (active_state or {}).get("pending_field"),
                        }
                    ),
                },
            ],
            temperature=0,
            max_tokens=8,
        )
        intent = completion.choices[0].message.content.strip().lower()
        return intent if intent in {"greeting", "inquiry", "high_intent", "lead_collection"} else None
    except Exception:
        return None


def detect_intent(message: str, active_state: Optional[AgentState] = None) -> str:
    text = message.lower()
    if active_state and active_state.get("pending_field"):
        return "lead_collection"

    llm_intent = _llm_intent(message, active_state)
    if llm_intent:
        return llm_intent

    high_intent_terms = [
        "sign up",
        "sign me up",
        "signup",
        "subscribe",
        "buy",
        "purchase",
        "start trial",
        "free trial",
        "book a demo",
        "talk to sales",
        "get started",
        "want to try",
        "ready",
        "pro plan for my",
    ]
    inquiry_terms = [
        "pricing",
        "price",
        "cost",
        "plan",
        "feature",
        "refund",
        "support",
        "4k",
        "captions",
        "videos",
        "resolution",
    ]
    greeting_terms = ["hi", "hello", "hey", "good morning", "good evening"]

    if any(term in text for term in high_intent_terms):
        return "high_intent"
    if any(term in text for term in inquiry_terms):
        return "inquiry"
    if any(re.search(rf"\b{re.escape(term)}\b", text) for term in greeting_terms):
        return "greeting"
    return "inquiry"


def _format_rag_answer(message: str, docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "AutoStream helps creators automate video editing with AI."

    primary = docs[0]
    if primary["id"] == "pricing":
        return (
            "AutoStream has two plans: Basic is $29/month for 10 videos/month at "
            "720p. Pro is $79/month with unlimited videos, 4K resolution, AI "
            "captions, and Pro support."
        )
    if primary["id"] == "refunds":
        return "AutoStream allows refunds within the first 7 days. After 7 days, refunds are not available."
    if primary["id"] == "support":
        return "AutoStream offers 24/7 support on the Pro plan only."

    return primary["content"]


def _extract_email(text: str) -> Optional[str]:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else None


def _extract_platform(text: str) -> Optional[str]:
    tokens = _tokenize(text)
    for key, label in PLATFORMS.items():
        if key in tokens:
            return label
    return None


def _extract_name(text: str) -> Optional[str]:
    cleaned = text.strip()
    patterns = [
        r"\bmy name is\s+([a-z][a-z .'-]{1,60})",
        r"\bi am\s+([a-z][a-z .'-]{1,60})",
        r"\bi'm\s+([a-z][a-z .'-]{1,60})",
        r"\bthis is\s+([a-z][a-z .'-]{1,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return _clean_name(match.group(1))

    if (
        EMAIL_RE.search(cleaned)
        or _extract_platform(cleaned)
        or len(cleaned.split()) > 4
        or any(char.isdigit() for char in cleaned)
    ):
        return None
    return _clean_name(cleaned)


def _clean_name(name: str) -> Optional[str]:
    name = re.split(r"\b(?:and|email|platform)\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
    name = re.sub(r"\s+", " ", name).strip(" .,-")
    if not name:
        return None
    return " ".join(part.capitalize() for part in name.split())


def _next_missing_field(lead: Dict[str, Optional[str]]) -> Optional[str]:
    for field in ("name", "email", "platform"):
        if not lead.get(field):
            return field
    return None


def _question_for(field: str) -> str:
    prompts = {
        "name": "Great, I can help with that. What is your name?",
        "email": "Thanks. What email should we use?",
        "platform": "Which creator platform do you use? For example, YouTube, Instagram, or TikTok.",
    }
    return prompts[field]


def _merge_lead_details(state: AgentState, message: str) -> None:
    lead = state["lead"]
    email = _extract_email(message)
    platform = _extract_platform(message)

    if email:
        lead["email"] = email
    if platform:
        lead["platform"] = platform

    pending = state.get("pending_field")
    if pending == "name" and not lead.get("name"):
        lead["name"] = _extract_name(message)
    elif pending == "email" and not email:
        return
    elif pending == "platform" and not platform:
        lead["platform"] = message.strip()
    elif not lead.get("name"):
        lead["name"] = _extract_name(message)


def _handle_lead_collection(state: AgentState, message: str) -> AgentState:
    previous_pending = state.get("pending_field")
    _merge_lead_details(state, message)
    missing = _next_missing_field(state["lead"])

    if missing:
        state["pending_field"] = missing
        if missing == "email" and previous_pending == "email" and not _extract_email(message):
            state["response"] = "Please share a valid email address so I can capture the lead."
        else:
            state["response"] = _question_for(missing)
        return state

    if not state.get("lead_captured"):
        lead = state["lead"]
        mock_lead_capture(lead["name"], lead["email"], lead["platform"])
        state["lead_captured"] = True

    state["pending_field"] = None
    state["intent"] = "lead_captured"
    state["response"] = (
        f"Thanks {state['lead']['name']}! I captured your interest in AutoStream "
        f"for {state['lead']['platform']}. Our team will follow up at {state['lead']['email']}."
    )
    return state


class AutoStreamAgent:
    def __init__(self) -> None:
        self.state = _empty_state()
        self.graph = self._build_graph()

    def reset(self) -> None:
        self.state = _empty_state()

    def respond(self, user_message: str) -> str:
        if not user_message.strip():
            return "Please send a message so I can help."

        if self.graph:
            result = self.graph.invoke({**self.state, "current_message": user_message})
            self.state = {key: result[key] for key in result if key != "current_message"}
        else:
            self.state = self._respond_without_langgraph(user_message)

        response = self.state["response"]
        self.state["history"].append({"user": user_message, "assistant": response})
        self.state["history"] = self.state["history"][-6:]
        return response

    def _respond_without_langgraph(self, user_message: str) -> AgentState:
        state = {**self.state, "current_message": user_message}
        state = self._classify_node(state)
        route = self._route_after_classification(state)
        if route == "greeting":
            return self._greeting_node(state)
        if route == "lead_collection":
            return self._lead_node(state)
        return self._rag_node(state)

    def _build_graph(self):
        if StateGraph is None:
            return None

        workflow = StateGraph(AgentState)
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("greeting", self._greeting_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("lead_collection", self._lead_node)
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "greeting": "greeting",
                "rag": "rag",
                "lead_collection": "lead_collection",
            },
        )
        workflow.add_edge("greeting", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("lead_collection", END)
        return workflow.compile()

    def _classify_node(self, state: AgentState) -> AgentState:
        intent = detect_intent(state["current_message"], state)
        return {**state, "intent": intent}

    def _route_after_classification(self, state: AgentState) -> str:
        if state["intent"] == "greeting":
            return "greeting"
        if state["intent"] in {"high_intent", "lead_collection"}:
            return "lead_collection"
        return "rag"

    def _greeting_node(self, state: AgentState) -> AgentState:
        return {
            **state,
            "response": "Hello! I can answer AutoStream pricing, feature, refund, and support questions.",
        }

    def _rag_node(self, state: AgentState) -> AgentState:
        docs = retrieve_documents(state["current_message"])
        return {
            **state,
            "retrieved_docs": docs,
            "response": _format_rag_answer(state["current_message"], docs),
        }

    def _lead_node(self, state: AgentState) -> AgentState:
        return _handle_lead_collection(state, state["current_message"])


agent = AutoStreamAgent()


def respond(user_msg: str) -> str:
    return agent.respond(user_msg)


def chat_loop() -> None:
    print("AutoStream Agent Ready (type quit to exit, reset to restart)")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        if user.lower() == "reset":
            agent.reset()
            print("Bot: Conversation reset.")
            continue
        print("Bot:", respond(user))
