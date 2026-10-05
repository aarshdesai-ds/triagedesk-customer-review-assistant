from chains import final_chain
from agent import ask


def print_report(report):
    print("\n========== REVIEW TRIAGE REPORT ==========")
    print(f"{'Product:':<13}{report['product']}")
    print(f"{'Reviewer:':<13}{report['reviewer'] or 'Unknown'}")
    print(f"{'Sentiment:':<13}{report['sentiment']}")
    print(f"{'Urgency:':<13}{report['urgency']}")
    print(f"{'Sources:':<13}{' '.join(report['sources'])}")

    print("\nSummary:")
    print(f"  {report['summary']}")

    print("\nIssues:")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  - {issue}")
    else:
        print("  None reported")

    print("\nRetrieved context:")
    print(f"  {report['policy']}")

    print(f"\n---------- Suggested reply ({report['reply_word_count']} words) ----------")
    print(report["reply"])
    if report["escalate"]:
        print("\n⚠ ESCALATE: negative review with high urgency. Send to a supervisor.")
    print("==========================================\n")

review_text = input("Agent: Please enter the customer review: \n")
report = final_chain.invoke({"review":review_text})    
print_report(report)
final_chain.get_graph().print_ascii()

product = report['product']
review = report['review']

history = []
print("Ask follow-up questions (type 'exit' to quit):")
while True:
    question = input("Agent: ")
    if question.strip().lower() == "exit":
        break
    else:
        if not question.strip():
            continue
        answer = ask(question, product, review, history)
        print("Assistant:", answer)
        history.append((question,answer))
