from langgraph.graph import StateGraph, START, END
from graph_state import TriageState
from typing import Literal
from graph_nodes import analyze, summarize, retrieve, build_report, positive_reply, negative_reply, mixed_reply, check_escalation

def route_by_sentiment(state: TriageState) -> Literal['positive', 'negative', 'mixed']:
    return state['analysis']['sentiment']

graph = StateGraph(TriageState)

graph.add_node("analyze", analyze)
graph.add_node("summarize", summarize)
graph.add_node("retrieve", retrieve)
graph.add_node("check_escalation", check_escalation)
graph.add_node("positive_reply", positive_reply)
graph.add_node("negative_reply", negative_reply)
graph.add_node("mixed_reply", mixed_reply)
graph.add_node("build_report", build_report)

graph.add_edge(START, "analyze")
graph.add_edge(START, "summarize")
graph.add_edge("analyze", "retrieve")
graph.add_edge("summarize", "retrieve")
graph.add_edge("retrieve", "check_escalation")
graph.add_edge("positive_reply", "build_report")
graph.add_edge("negative_reply", "build_report")
graph.add_edge("mixed_reply", "build_report")

graph.add_conditional_edges("check_escalation",route_by_sentiment, {
    "positive": "positive_reply",
    "negative": "negative_reply",
    "mixed": "mixed_reply"
})

graph.add_edge("build_report", END)

workflow = graph.compile()


if __name__ == "__main__":
    import sys

    print("--- nodes (free) ---")
    print(sorted(workflow.get_graph().nodes))

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        review = ("The headphones stopped charging after a week and support has not "
                  "replied to two emails. Really disappointed. - Arjun")
        config = {"run_name": "triagedesk-graph-test",
                  "tags": ["triagedesk", "v2", "test"],
                  "metadata": {"source": "triage_graph main", "expect": "negative"}}
        result = workflow.invoke({"review": review}, config=config)
        report = result["report"]
        print("--- one review through the graph (uses the model) ---")
        for key in ["product", "sentiment", "urgency", "escalate",
                    "escalated_by", "hazard_words", "sources", "reply_word_count"]:
            print(f"  {key:<17}{report[key]}")
        print("reply:", report["reply"])