from langgraph.graph import StateGraph, START, END
from graph_state import TriageState
from storage import checkpointer
from typing import Literal
from graph_nodes import analyze, summarize, retrieve, build_report, positive_reply, negative_reply, mixed_reply, check_escalation, supervisor_review, safety_reply

def route_reply(state: TriageState) -> Literal["safety", 'positive', 'negative', 'mixed']:
    if state['hazard_words']:
        return "safety"
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
    graph.add_node("safety_reply", safety_reply)
    graph.add_node("supervisor_review", supervisor_review)
    graph.add_node("build_report", build_report)

    graph.add_edge(START, "analyze")
    graph.add_edge(START, "summarize")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("summarize", "retrieve")
    graph.add_edge("retrieve", "check_escalation")


    graph.add_conditional_edges("check_escalation",route_reply, {
    "safety": "safety_reply",   
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

    graph.add_conditional_edges("safety_reply", route_after_reply,{
        "supervisor_review": "supervisor_review",
        "build_report": "build_report"
    })
    

    graph.add_edge("supervisor_review", "build_report")
    graph.add_edge("build_report", END)

    return graph.compile(checkpointer=checkpointer)

workflow = build_graph(checkpointer=checkpointer)