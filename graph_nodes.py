from chains import analysis_chain, summary_chain, retriever, format_docs, unique_sources, positive_chain, mixed_chain, negative_chain
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

if __name__ == "__main__":
    import sys

    print("--- check_escalation (free) ---")
    cases = [
        # (review, sentiment, urgency, expected escalate, expected escalated_by)
        ("Great monitor but I saw a spark at the power port.", "mixed",    "high",   True,  "rules"),
        ("The charger sparked and now it is dead. Furious.",    "negative", "high",   True,  "both"),
        ("Stopped working after a week, support ignores me.",   "negative", "high",   True,  "urgency"),
        ("blendr is gud but jar lid broke.. pls replace asap",  "mixed",    "high",   False, "none"),
        ("Love it, works perfectly.",                           "positive", "none",   False, "none"),
    ]
    for review, sentiment, urgency, want_escalate, want_by in cases:
        state = {"review": review, "analysis": {"sentiment": sentiment, "urgency": urgency}}
        out = check_escalation(state)
        ok = out["escalate"] == want_escalate and out["escalated_by"] == want_by
        print(f"{'ok  ' if ok else 'FAIL'} {sentiment:<8} {urgency:<6} -> {out}")
    print("\n--- supervisor_review (free, tiny graph) ---")
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from graph_state import TriageState

    mini = StateGraph(TriageState)
    mini.add_node("supervisor_review", supervisor_review)
    mini.add_edge(START, "supervisor_review")
    mini.add_edge("supervisor_review", END)
    mini_graph = mini.compile(checkpointer=InMemorySaver())

    start = {"review": "spark at the power port", "escalated_by": "rules",
             "hazard_words": ["spark"], "reply": "Please send photos."}
    for thread, answer, want_decision, want_reply in [
        ("t-approve", " Approve ", "approved", "Please send photos."),
        ("t-edit", "Please stop using it.", "edited", "Please stop using it."),
    ]:
        config = {"configurable": {"thread_id": thread}}
        paused = mini_graph.invoke(start, config=config)
        payload = paused["__interrupt__"][0].value
        done = mini_graph.invoke(Command(resume=answer), config=config)
        ok = done["review_decision"] == want_decision and done["reply"] == want_reply
        print(f"{'ok  ' if ok else 'FAIL'} {thread:<10} payload words {payload['hazard_words']} -> "
              f"{done['review_decision']}, reply {done['reply']!r}")

    print("\n--- build_report default (free) ---")
    base = {"review": "x", "analysis": {"product": "p", "sentiment": "positive", "urgency": "none",
            "issues": [], "reviewer": ""}, "summary": "", "policy": "", "sources": [], "reply": "Thanks!",
            "escalate": False, "escalated_by": "none", "hazard_words": []}
    print("ok  " if build_report(base)["report"]["review_decision"] == "not_needed" else "FAIL", "missing key -> not_needed")
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("--- one review through the nodes (uses the model) ---")
        state = {"review": "I love this monitor, the picture is great. But yesterday I saw a "
                           "small spark at the power port when I plugged it in. - Leela"}
        for node in [analyze, summarize, retrieve, check_escalation, mixed_reply, build_report]:
            state.update(node(state))
        report = state["report"]
        for key in ["product", "sentiment", "urgency", "sources",
                    "escalate", "escalated_by", "hazard_words"]:
            print(f"  {key:<14}{report[key]}")
        print("reply:", report["reply"])