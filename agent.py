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
GREETING_RE = re.compile(r"\b(?:hi|hello|hey|good morning|good evening)\b", re.IGNORECASE)
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
    greeted: bool


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
        "greeted": False,
    }


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _knowledge_documents() -> List[Dict[str, Any]]:
    documents = KB.get("documents")
    if documents:
        return documents

    return [
        {
            "id": "pricing",
            "title": "Pricing and features",
            "content": KB.get("pricing_text", ""),
            "topics": ["pricing", "plans", "features"],
        },
        {
            "id": "refunds",
            "title": "Refund policy",
            "content": KB.get("refund_policy", ""),
            "topics": ["refund", "policy"],
        },
        {
            "id": "support",
            "title": "Support policy",
            "content": KB.get("support_policy", ""),
            "topics": ["support", "policy"],
        },
    ]


def _expand_query_tokens(query_tokens: set) -> set:
    boosts = {
        "price": {"pricing", "plan", "cost", "monthly", "basic", "pro"},
        "pricing": {"pricing", "plan", "cost", "monthly", "basic", "pro"},
        "cost": {"pricing", "price", "plan", "monthly", "basic", "pro"},
        "plan": {"pricing", "price", "cost", "monthly", "basic", "pro"},
        "basic": {"pricing", "price", "plan", "monthly", "720p", "videos"},
        "pro": {"pricing", "price", "plan", "monthly", "4k", "captions", "support"},
        "refund": {"refund", "policy", "days"},
        "support": {"support", "24", "7", "pro"},
        "feature": {"feature", "video", "caption", "resolution", "4k", "720p"},
        "features": {"feature", "video", "caption", "resolution", "4k", "720p"},
        "trial": {"signup", "lead", "demo", "pricing", "pro"},
    }
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        expanded_tokens.update(boosts.get(token, set()))
    return expanded_tokens


def retrieve_documents(query: str, limit: int = 2) -> List[Dict[str, Any]]:
    """Local retriever using keyword overlap over the JSON knowledge base."""
    query_tokens = _expand_query_tokens(_tokenize(query))
    ranked = []

    for doc in _knowledge_documents():
        parts = [doc.get("title", ""), doc.get("content", "")]
        topics = doc.get("topics") or []
        if topics:
            parts.append(" ".join(topics))
        haystack = " ".join(parts)
        doc_tokens = _tokenize(haystack)
        score = len(query_tokens & doc_tokens)
        ranked.append({**doc, "score": score})

    ranked.sort(key=lambda item: (item["score"], len(item.get("content", ""))), reverse=True)
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
                        "Use inquiry for mixed greeting-plus-question messages. "
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
        valid = {"greeting", "inquiry", "high_intent", "lead_collection"}
        return intent if intent in valid else None
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
        "ready to start",
        "ready to buy",
        "i'm interested",
        "interested in signing up",
    ]
    inquiry_terms = [
        "pricing",
        "price",
        "cost",
        "plan",
        "feature",
        "features",
        "refund",
        "support",
        "4k",
        "captions",
        "videos",
        "resolution",
        "what do you do",
        "how does it work",
    ]

    has_greeting = bool(GREETING_RE.search(text))
    has_high_intent = any(term in text for term in high_intent_terms)
    has_inquiry = any(term in text for term in inquiry_terms)

    if has_high_intent:
        return "high_intent"
    if has_inquiry:
        return "inquiry"
    if has_greeting:
        return "greeting"
    return "inquiry"


def _message_has_greeting(message: str) -> bool:
    return bool(GREETING_RE.search(message))


def _extract_requested_topics(message: str) -> List[str]:
    text = message.lower()
    topics = []
    if any(term in text for term in ["price", "pricing", "cost", "plan", "plans"]):
        topics.append("pricing")
    if any(term in text for term in ["feature", "features", "4k", "captions", "resolution", "videos"]):
        topics.append("features")
    if "refund" in text:
        topics.append("refund")
    if "support" in text:
        topics.append("support")
    if any(term in text for term in ["what do you do", "how does it work", "product", "autostream"]):
        topics.append("overview")
    return topics


def _summarize_doc(doc: Dict[str, Any], message: str) -> str:
    doc_id = doc.get("id")
    if doc_id == "pricing":
        requested_topics = _extract_requested_topics(message)
        include_features = "features" in requested_topics or "pricing" not in requested_topics
        pricing = "Basic is $29/month for 10 videos/month at 720p. Pro is $79/month with unlimited videos."
        if include_features:
            pricing += " Pro also includes 4K exports, AI captions, and 24/7 support."
        return pricing
    if doc_id == "refunds":
        return "Refunds are available only within the first 7 days."
    if doc_id == "support":
        return "24/7 support is included on the Pro plan only."
    if doc_id == "product_overview":
        return (
            "AutoStream helps creators automate video editing, add AI captions, and "
            "prepare polished videos for social platforms."
        )
    return doc.get("content", "")


def _format_rag_answer(message: str, docs: List[Dict[str, Any]], greeted: bool = False) -> str:
    if not docs:
        answer = "AutoStream helps creators automate video editing with AI."
        return f"Hi! {answer}" if greeted else answer

    snippets = []
    seen = set()
    for doc in docs:
        snippet = _summarize_doc(doc, message)
        if snippet and snippet not in seen:
            snippets.append(snippet)
            seen.add(snippet)

    if not snippets:
        snippets = [docs[0].get("content", "AutoStream helps creators automate video editing with AI.")]

    requested_topics = _extract_requested_topics(message)
    asked_about_product = "overview" in requested_topics
    answer_parts = []
    if greeted:
        answer_parts.append("Hi!")
    answer_parts.append(snippets[0])

    if len(snippets) > 1 and requested_topics:
        answer_parts.append(snippets[1])
    elif asked_about_product and len(snippets) == 1:
        answer_parts.append("I can also walk you through plans, support, or refund policy.")

    return " ".join(part.strip() for part in answer_parts if part).strip()


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
    lowered = cleaned.lower()
    blocked_phrases = [
        "i want to",
        "get started",
        "sign me up",
        "sign up",
        "start trial",
        "free trial",
        "book a demo",
        "talk to sales",
    ]
    if any(phrase in lowered for phrase in blocked_phrases):
        return None
    patterns = [
        r"\bmy name is\s+([a-z][a-z .'-]{1,60})",
        r"\bi am\s+([a-z][a-z .'-]{1,60})",
        r"\bi'm\s+([a-z][a-z .'-]{1,60})",
        r"\bthis is\s+([a-z][a-z .'-]{1,60})",
        r"\bname[:\s]+([a-z][a-z .'-]{1,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return _clean_name(match.group(1))

    if (
        EMAIL_RE.search(cleaned)
        or _extract_platform(cleaned)
        or len(cleaned.split()) > 5
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
        "name": "I can help you get started. What is your name?",
        "email": "Got it. What email should we use?",
        "platform": "Which creator platform do you use most? For example, YouTube, Instagram, or TikTok.",
    }
    return prompts[field]


def _merge_lead_details(state: AgentState, message: str) -> None:
    lead = state["lead"]
    email = _extract_email(message)
    platform = _extract_platform(message)
    name = _extract_name(message)

    if email:
        lead["email"] = email
    if platform:
        lead["platform"] = platform
    if name and not lead.get("name"):
        lead["name"] = name

    pending = state.get("pending_field")
    if pending == "platform" and not platform and message.strip():
        lead["platform"] = message.strip()


def _lead_intro(state: AgentState, message: str) -> str:
    lead = state["lead"]
    requested_topics = _extract_requested_topics(message)
    answer_parts = []

    if "pricing" in requested_topics or "features" in requested_topics:
        docs = retrieve_documents(message, limit=1)
        answer_parts.append(_format_rag_answer(message, docs, greeted=_message_has_greeting(message)))
    elif _message_has_greeting(message):
        answer_parts.append("Hi!")

    if lead.get("name") and not lead.get("email"):
        answer_parts.append(f"Happy to help, {lead['name']}. What email should we use?")
    else:
        answer_parts.append(_question_for(state["pending_field"]))
    return " ".join(part.strip() for part in answer_parts if part).strip()


def _handle_lead_collection(state: AgentState, message: str) -> AgentState:
    previous_pending = state.get("pending_field")
    _merge_lead_details(state, message)
    missing = _next_missing_field(state["lead"])

    if missing:
        state["pending_field"] = missing
        if previous_pending is None:
            state["response"] = _lead_intro(state, message)
            return state
        if missing == "email" and previous_pending == "email" and not _extract_email(message):
            state["response"] = "Please share a valid email address so I can capture the lead."
            return state
        if missing == "platform" and previous_pending == "platform" and not message.strip():
            state["response"] = _question_for("platform")
            return state
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
        response = "Hello! I can answer AutoStream pricing, features, refunds, and support questions."
        return {**state, "response": response, "greeted": True}

    def _rag_node(self, state: AgentState) -> AgentState:
        greeted = _message_has_greeting(state["current_message"]) and not state.get("greeted", False)
        docs = retrieve_documents(state["current_message"])
        return {
            **state,
            "retrieved_docs": docs,
            "response": _format_rag_answer(state["current_message"], docs, greeted=greeted),
            "greeted": state.get("greeted", False) or greeted,
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
