from langgraph.graph import StateGraph, START, END
from graph_state import TriageState
from storage import checkpointer
from typing import Literal
from graph_nodes import analyze, summarize, retrieve, build_report, positive_reply, negative_reply, mixed_reply, check_escalation, supervisor_review

def route_by_sentiment(state: TriageState) -> Literal['positive', 'negative', 'mixed']:
    return state['analysis']['sentiment']

def route_after_reply(state: TriageState) -> Literal['supervisor_review', 'build_report']:
    if state['escalate']:
        return "supervisor_review"
    return "build_report"

def build_graph(checkpointer = None):

    graph = StateGraph(TriageState)

    graph.add_node("analyze", analyze)
    graph.add_node("summarize", summarize)
    graph.add_node("retrieve", retrieve)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("positive_reply", positive_reply)
    graph.add_node("negative_reply", negative_reply)
    graph.add_node("mixed_reply", mixed_reply)
    graph.add_node("supervisor_review", supervisor_review)
    graph.add_node("build_report", build_report)

    graph.add_edge(START, "analyze")
    graph.add_edge(START, "summarize")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("summarize", "retrieve")
    graph.add_edge("retrieve", "check_escalation")


    graph.add_conditional_edges("check_escalation",route_by_sentiment, {
    "positive": "positive_reply",
    "negative": "negative_reply",
    "mixed": "mixed_reply"
    })

    graph.add_conditional_edges("positive_reply", route_after_reply,{
    "supervisor_review": "supervisor_review",
    "build_report": "build_report"
    })

    graph.add_conditional_edges("negative_reply", route_after_reply,{
    "supervisor_review": "supervisor_review",
    "build_report": "build_report"
    })

    graph.add_conditional_edges("mixed_reply", route_after_reply,{
    "supervisor_review": "supervisor_review",
    "build_report": "build_report"
    })

    graph.add_edge("supervisor_review", "build_report")
    graph.add_edge("build_report", END)

    return graph.compile(checkpointer=checkpointer)

workflow = build_graph(checkpointer=checkpointer)

if __name__ == "__main__":
    import sys
    from uuid import uuid4
    from langgraph.types import Command
    from storage import config_for

    print("--- nodes (free) ---")
    print(sorted(workflow.get_graph().nodes))

    print("\n--- route_after_reply (free) ---")
    for escalate, want in [(True, "supervisor_review"), (False, "build_report")]:
        got = route_after_reply({"escalate": escalate})
        print(f"{'ok  ' if got == want else 'FAIL'} escalate={escalate!s:<5} -> {got}")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- Leela through the graph, pause, approve (uses the model) ---")
        review = ("I love this monitor, the picture is great. But yesterday I saw a "
                  "small spark at the power port when I plugged it in. - Leela")
        config = {**config_for(f"test-{uuid4().hex[:6]}"),
                  "run_name": "triagedesk-hitl-test", "tags": ["triagedesk", "v2", "hitl"]}
        result = workflow.invoke({"review": review}, config=config)
        if result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            print("paused:", payload["escalated_by"], payload["hazard_words"])
            print("draft:", payload["draft"][:120], "...")
            result = workflow.invoke(Command(resume="approve"), config=config)
        else:
            print("FAIL did not pause")
        report = result["report"]
        print("review_decision:", report["review_decision"])