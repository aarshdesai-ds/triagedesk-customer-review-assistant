import json
import csv
from chains import final_chain
from collections import Counter

PLACEHOLDER_NAMES = ["", "unknown", "anonymous", "none", "n/a", "not provided"]

REVIEWS_PATH = "eval/reviews.jsonl"
RESULTS_PATH = "eval/results.csv"

rows = []
with open(REVIEWS_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

def evaluate_row(row):
    result = {**row,
              "got_product": "",
              "got_sentiment":"",
              "got_urgency": "",
              "got_reviewer": "",
              "got_escalate": None,
              "got_sources":[],
              "got_issues": [],
              "got_reply": "",
              "sentiment_ok": None,
              "product_ok": None,
              "escalate_ok": None,
              "source_ok": None, 
              "reviewer_ok": None,
              "error": ""}
    try:
        report = final_chain.invoke({"review":row['review']})
        product = report['product'] or ""
        sentiment = report['sentiment']
        urgency = report['urgency']
        reviewer = report['reviewer'] or ""
        sources = report['sources']
        escalate = report['escalate']
        issues = report['issues'] or []
        reply = report['reply']

        acceptable_sentiments = row["acceptable_sentiments"]   
        expected_product = row["expected_product"]
        expected_escalate = row["expected_escalate"]    
        acceptable_sources = row['acceptable_sources']     
        expected_reviewer = row["expected_reviewer"]

        if not expected_product:
            product_ok = None
        else:
            product_ok = expected_product in product.lower()

        sentiment_ok = sentiment in acceptable_sentiments

        if expected_escalate is None:
            escalate_ok = None
        else:
            escalate_ok = expected_escalate == escalate

        if acceptable_sources is None:
            source_ok = None
        else:
            source_ok = any(source in sources for source in acceptable_sources)

        reviewer_text = reviewer.strip().lower()
        if not expected_reviewer:
            reviewer_ok = reviewer_text in PLACEHOLDER_NAMES
        else:
            reviewer_ok = expected_reviewer.lower() in reviewer_text

        result['got_product'] = product
        result['got_sentiment'] = sentiment
        result['got_urgency'] = urgency
        result['got_reviewer'] = reviewer
        result['got_escalate'] = escalate
        result['got_sources'] = sources
        result['got_issues'] = issues
        result['got_reply'] = reply
        result['product_ok'] = product_ok
        result['sentiment_ok'] = sentiment_ok
        result['escalate_ok'] = escalate_ok
        result['reviewer_ok'] = reviewer_ok
        result['source_ok'] = source_ok

    except Exception as e:
        result['error'] = str(e)

    return result

def summarise_escalate(results):
    results = [row for row in results if not row['error']]
    checked = [row for row in results if row['escalate_ok'] is not None]
    caught = [row for row in checked if row['expected_escalate'] and row['got_escalate']]
    missed = [row for row in checked if row['expected_escalate'] and not row['got_escalate']]
    false_alarms = [row for row in checked if not row['expected_escalate'] and row['got_escalate']]
    correct = [row for row in checked if not row['expected_escalate'] and not row['got_escalate']]

    should_escalate = len(caught) + len(missed)
    should_not_escalate = len(false_alarms) + len(correct)

    return {"caught":caught, "missed":missed, "false_alarms":false_alarms, "correct":correct, "should_escalate":should_escalate, "should_not_escalate":should_not_escalate}

def split(results, key):
    results = [row for row in results if not row['error']]
    checked = [row for row in results if row[key] is not None]
    hits = [row for row in checked if row[key]]
    misses = [row for row in checked if not row[key]]

    return checked, hits, misses

def summarise_quality(results):
    ok = [row for row in results if not row['error']]
    return {"sentiment": split(ok, "sentiment_ok"),
             "product":   split(ok, "product_ok"),
             "source":    split(ok, "source_ok"),
             "reviewer":  split(ok, "reviewer_ok")}

def summarise_groups(results):
    results = [row for row in results if not row['error']]
    groups = {}
    for row in results:
        groups.setdefault(row['group'],[])
        groups[row['group']].append(row)
    return groups

def save_csv(results, path):
    if not results:
        return 
    fieldnames = list(results[0].keys())
    csv_rows = []
    for row in results:
        csv_rows.append({**row, "acceptable_sentiments": " | ".join(row['acceptable_sentiments']) , "got_sources": " | ".join(row['got_sources']), "acceptable_sources":json.dumps(row['acceptable_sources']),  "got_issues": " | ".join(row['got_issues'])})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f = f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    print(f"Loaded {len(rows)} reviews")
    rows_to_run = rows

    results = []
    for number, row in enumerate(rows_to_run, start=1):
        result = evaluate_row(row)
        results.append(result)
        if result["error"]:
            status = "ERROR"
        else:
            checks = [result["sentiment_ok"], result["product_ok"], result["escalate_ok"],
                      result["source_ok"], result["reviewer_ok"]]
            status = "FAIL" if False in checks else "PASS"
        print(f"[{number}/{len(rows_to_run)}] {result['id']}  {result['group']:<18} {status}  "
              f"({result['got_sentiment']}, urgency {result['got_urgency']})")

    save_csv(results, RESULTS_PATH)
    print(f"\nSaved {len(results)} rows to {RESULTS_PATH}")

    escalation = summarise_escalate(results)
    quality = summarise_quality(results)
    errors = [row for row in results if row["error"]]

    print("\nESCALATION")
    print(f"Caught:        {len(escalation['caught'])} of {escalation['should_escalate']}")
    print(f"Missed:        {len(escalation['missed'])} of {escalation['should_escalate']}")
    print(f"False alarms:  {len(escalation['false_alarms'])} of {escalation['should_not_escalate']}")
    for row in escalation["missed"]:
        print(f"\n  MISSED {row['id']} (sentiment {row['got_sentiment']}, urgency {row['got_urgency']}): {row['review']}")
    for row in escalation["false_alarms"]:
        print(f"  FALSE ALARM {row['id']} (sentiment {row['got_sentiment']}, urgency {row['got_urgency']})")

    details = {
        "sentiment": ("got_sentiment", "acceptable_sentiments"),
        "product": ("got_product", "expected_product"),
        "source": ("got_sources", "acceptable_sources"),
        "reviewer": ("got_reviewer", "expected_reviewer"),
    }
    for name, (got_key, expected_key) in details.items():
        checked, hits, misses = quality[name]
        print(f"\n{name.upper()}")
        print(f"Correct:  {len(hits)} of {len(checked)} checked")
        for row in misses:
            print(f"  {row['id']}: got {row[got_key]!r}, expected {row[expected_key]!r}")
    
    print("\nNO ISSUES LISTED ON A NON-POSITIVE REVIEW")
    suspects = [row for row in results
                if not row["error"] and row["got_sentiment"] != "positive" and not row["got_issues"]]
    print(f"Rows:  {len(suspects)}")
    for row in suspects:
        print(f"  {row['id']} ({row['got_sentiment']}): sources {row['got_sources']}")

    print("\nBY GROUP")
    for group, group_rows in summarise_groups(results).items():
        sentiment_passed = [row for row in group_rows if row["sentiment_ok"]]
        escalate_checked = [row for row in group_rows if row["escalate_ok"] is not None]
        escalate_passed = [row for row in escalate_checked if row["escalate_ok"]]
        print(f"{group:<18} {len(group_rows)} rows   sentiment {len(sentiment_passed)}/{len(group_rows)}   "
              f"escalation {len(escalate_passed)}/{len(escalate_checked)}")

    print("\nERRORS")
    print(f"Failed calls:  {len(errors)} of {len(results)}")
    for row in errors:
        print(f"  {row['id']}: {row['error'][:300]}")