# AutoStream Social-to-Lead Agent

Conversational AI agent for a fictional SaaS product, AutoStream. The agent answers product questions from a local knowledge base, detects high-intent users, collects lead details, and calls a mock lead-capture tool only after all required fields are available.

## Features

- Intent detection for greetings, product/pricing inquiries, and high-intent signup messages
- Local RAG-style retrieval from `knowledge_base.json`
- Stateful 5-6 turn conversations with recent message memory
- Lead collection for name, email, and creator platform
- Mock backend tool call through `mock_lead_capture(name, email, platform)`
- Optional GPT-4o-mini intent classification, with a deterministic fallback for local demos
- LangGraph orchestration when `langgraph` is installed

## How To Run Locally

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Start the CLI chat.

```bash
python app.py
```

Optional LLM-assisted intent classification:

```bash
set OPENAI_API_KEY=your_api_key
set USE_LLM_INTENT=true
set OPENAI_MODEL=gpt-4o-mini
python app.py
```

Try this demo flow:

```text
You: Hi, tell me about your pricing.
You: That sounds good, I want to try the Pro plan for my YouTube channel.
You: Priya Sharma
You: priya@example.com
```

The final turn should print `Lead captured successfully: Priya Sharma, priya@example.com, YouTube`.

## Architecture Explanation

The agent uses LangGraph because the assignment is a stateful agent workflow, not a single-turn chatbot. The graph has four nodes: `classify`, `greeting`, `rag`, and `lead_collection`. The `classify` node determines whether the user is greeting, asking a product question, or showing buying intent. Product questions route to the RAG node, which retrieves the most relevant entries from `knowledge_base.json` using local keyword overlap and then formats a grounded answer from the retrieved document. High-intent messages route to lead collection.

State is stored in an `AgentState` dictionary containing recent conversation history, current intent, pending lead field, captured lead values, retrieved documents, and the latest response. This lets the agent remember what it asked for across multiple turns and prevents the lead-capture tool from firing early. The tool call happens only when `name`, `email`, and `platform` are all populated. If `USE_LLM_INTENT=true` and `OPENAI_API_KEY` is set, GPT-4o-mini classifies intent; otherwise the agent uses deterministic rules so the demo remains easy to run offline.

## WhatsApp Deployment With Webhooks

To deploy this agent on WhatsApp, I would connect a WhatsApp Business Platform number to a backend service such as FastAPI or Flask. Meta would send inbound WhatsApp messages to a webhook endpoint like `POST /webhooks/whatsapp`. The endpoint would verify Meta's webhook signature, normalize the sender phone number and message text, and load the matching conversation state from Redis, Postgres, or another persistent store.

The backend would pass the message into `AutoStreamAgent.respond(...)`, save the updated state, and send the response back through the WhatsApp Cloud API `/messages` endpoint. The lead-capture tool could be replaced with a CRM API call or database insert. For production, I would add retry handling, idempotency keys for webhook events, PII-safe logging, rate limits, and a handoff path when a user asks for a human sales rep.
