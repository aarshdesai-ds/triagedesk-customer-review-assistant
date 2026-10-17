import sys
from uuid import uuid4
from langgraph.types import Command
from triage_graph import workflow
from storage import config_for

def make_config():
    config = config_for(f"cli-{uuid4().hex[:6]}")
    config = {**config, "run_name": "triagedesk-cli", "tags": ["triagedesk", "v2", "cli"], "metadata": {"source": "run_review"}}
    return config

def show_payload(payload):
    words = ", ".join(payload['hazard_words'])
    print(f"ESCALATED ({payload['escalated_by']}) hazard_words: {words}")
    print(f"Review: {payload['review']}")
    print(f"Draft reply:\n{payload['draft']}")

def ask_supervisor():
    print("Type 'approve' to send as is, or type a replacement reply:")
    while True:
        answer = input("> ").strip()
        if answer == "":
            print("Please type 'approve' or a reply.")
            continue
        return answer

def run(review, auto_answer = None):
    config = make_config()
    result = workflow.invoke({"review":review}, config=config)
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        if not auto_answer:
            show_payload(payload)
        answer = auto_answer or ask_supervisor()

        result = workflow.invoke(Command(resume=answer), config=config)
    return result['report']

def print_report(report):
    for key in ["product", "sentiment", "urgency", "escalate", "escalated_by", "hazard_words", "review_decision"]:
        print(f"{key:<16}{report[key]}")
    print(f"reply:\n{report['reply']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])
    else:
        review = input("Paste the review: ").strip()

    report = run(review)
    print("\n--- report ---")
    print_report(report)
