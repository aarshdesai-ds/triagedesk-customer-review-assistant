from chains import analysis_chain, summary_chain, retriever, format_docs, unique_sources, positive_chain, mixed_chain, negative_chain, safety_chain
import re
from graph_state import TriageState
from langgraph.types import interrupt

HAZARD_PATTERN = re.compile(
    r"\b(spark\w*|smok\w*|burn\w*|melt\w*|overheat\w*|scorch\w*|electric shock|shocked"
    r"|caught fire|on fire|fire hazard|exploded?)\b",
    re.IGNORECASE,
)

def analyze(state: TriageState):
    analysis = analysis_chain.invoke({"review":state['review']})
    return {"analysis": analysis.model_dump()}

def summarize(state: TriageState):
    summary = summary_chain.invoke({"review":state['review']})
    return {'summary': summary}

def retrieve(state: TriageState):
    analysis = state['analysis']
    product = analysis['product']
    issues = analysis['issues'] or []
    if issues:
        query = f"{product}: {', '.join(issues)}. {state['summary']}"
    else:
        query = "product review loyalty points"

    docs = retriever.invoke(query)

    return {"query": query, "policy": format_docs(docs), "sources": unique_sources(docs)}

def check_escalation(state: TriageState):
    matches = HAZARD_PATTERN.findall(state['review'])
    matches = [match.lower() for match in matches]
    words = sorted(set(matches))
    by_rules = len(words) > 0
    by_urgency = state['analysis']['urgency'] == "high" and state['analysis']['sentiment'] == "negative"

    if by_rules and by_urgency:
        escalated_by = "both"
    elif by_rules:
        escalated_by = "rules"
    elif by_urgency:
        escalated_by = "urgency"
    else:
        escalated_by = "none"

    return {"hazard_words": words, "escalate": by_urgency or by_rules, "escalated_by": escalated_by}

def positive_reply(state: TriageState):
    reply = positive_chain.invoke({"review": state['review'], "policy": state['policy']})
    return {"reply": reply}

def negative_reply(state: TriageState):
    reply = negative_chain.invoke({"review": state['review'], "policy": state['policy']})
    return {"reply": reply}

def mixed_reply(state: TriageState):
    reply = mixed_chain.invoke({"review": state['review'], "policy": state['policy']})
    return {"reply": reply}

def safety_reply(state: TriageState):
    reply = safety_chain.invoke({"review": state['review'], "policy": state['policy']})
    return {"reply": reply}

def supervisor_review(state: TriageState):
    payload = {'review': state['review'], "escalated_by": state['escalated_by'], "hazard_words":state['hazard_words'], "draft": state["reply"]}
    answer = interrupt(payload).strip()
    if answer.lower() == "approve":
        return {"review_decision": "approved"}
    else:
        return {'reply': answer, "review_decision":"edited"}


def build_report(state: TriageState):
    analysis = state['analysis']
    review = state['review']
    sentiment = analysis['sentiment']
    product = analysis['product']
    urgency = analysis['urgency']
    issues = analysis['issues']
    reviewer = analysis['reviewer']
    summary = state['summary']
    policy = state['policy']
    sources = state['sources']
    reply = state['reply']
    reply_word_count = len(reply.split())
    escalate = state['escalate']
    escalated_by = state['escalated_by']
    hazard_words = state['hazard_words']
    review_decision = state.get("review_decision", "not_needed")

    report = {"review": review, "product":product, "sentiment":sentiment, "urgency": urgency, "issues":issues, "reviewer": reviewer, "summary": summary, "policy": policy, "sources": sources, "reply": reply, "reply_word_count": reply_word_count, "escalate": escalate, "escalated_by": escalated_by, "hazard_words": hazard_words, "review_decision": review_decision}
    return {"report": report}