import json
import re
from tools import mock_lead_capture

with open('knowledge_base.json', 'r') as f:
    KB = json.load(f)

state = {
    'history': [],
    'intent': None,
    'lead': {'name': None, 'email': None, 'platform': None}
}


def detect_intent(msg):
    m = msg.lower()
    if any(x in m for x in ['hi', 'hello', 'hey']):
        return 'greeting'
    if any(x in m for x in ['pricing', 'price', 'plan', 'feature', 'refund', 'support']):
        return 'inquiry'
    if any(x in m for x in ['want to try', 'sign up', 'subscribe', 'buy', 'interested', 'pro plan']):
        return 'high_intent'
    return 'inquiry'


def retrieve_answer(msg):
    m = msg.lower()
    if 'price' in m or 'pricing' in m or 'plan' in m:
        return KB['pricing_text']
    if 'refund' in m:
        return KB['refund_policy']
    if 'support' in m:
        return KB['support_policy']
    return 'AutoStream helps creators automate video editing with AI.'


def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None


def collect_lead(user_msg):
    lead = state['lead']
    if not lead['name']:
        lead['name'] = user_msg.strip()
        return 'Please share your email.'
    if not lead['email']:
        email = extract_email(user_msg)
        if email:
            lead['email'] = email
            return 'Which creator platform do you use? (YouTube/Instagram/etc.)'
        return 'Please enter a valid email.'
    if not lead['platform']:
        lead['platform'] = user_msg.strip()
        mock_lead_capture(lead['name'], lead['email'], lead['platform'])
        return 'Thanks! Your lead has been captured successfully.'
    

def respond(user_msg):
    if state['intent'] == 'collecting_lead':
        return collect_lead(user_msg)

    intent = detect_intent(user_msg)
    state['intent'] = intent

    if intent == 'greeting':
        return 'Hello! Ask me about AutoStream pricing or features.'

    if intent == 'inquiry':
        return retrieve_answer(user_msg)

    if intent == 'high_intent':
        state['intent'] = 'collecting_lead'
        return 'Awesome! To get started, please share your name.'


def chat_loop():
    print('AutoStream Agent Ready (type quit to exit)')
    while True:
        user = input('You: ')
        if user.lower() == 'quit':
            break
        reply = respond(user)
        print('Bot:', reply)