import json
import csv
import re
from agent import ask_detailed

NOT_COVERED_SENTENCE = "the knowledge base does not cover this"

rows = []
with open("eval/followup.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))


def evaluate_row(row):
    result = {**row, 
              "tools_used": [],
              "answer": "",
              "tools_ok": None,
              "fact_ok": None,
              "not_covered_ok": None,
              "error": ""}
    try:
        out = ask_detailed(row['question'] , row['product'], row['review'], [])
        tools_used = out['tools_used']
        answer = out['answer']
        answer_text = answer.replace(",","").lower()

        acceptable = row["acceptable_tool_sets"]
        if acceptable is None:
            tools_ok = None
        else:
            used = sorted(set(tools_used))
            allowed = [sorted(one_set) for one_set in acceptable]
            tools_ok = used in allowed

        expected_facts = row['expected_facts']
        if expected_facts is None:
            fact_ok = None
        else:
            fact_ok = all(any(alt in answer_text for alt in group) for group in expected_facts)

        must_say_not_covered = row['must_say_not_covered']
        if not must_say_not_covered:
            not_covered_ok = None
        else:
            not_covered_ok = NOT_COVERED_SENTENCE in answer_text and len(tools_used) > 0

        result['tools_used'] = tools_used
        result['answer'] = answer
        result['tools_ok'] = tools_ok
        result['fact_ok'] = fact_ok
        result['not_covered_ok'] = not_covered_ok

    except Exception as e:
        result['error'] = str(e)
    return result

def summarise_results(results):
    results = [row for row in results if not row['error']]
    tool_checked = [row for row in results if row['tools_ok'] is not None]
    tool_hits = [row for row in tool_checked if row['tools_ok']]
    tool_misses = [row for row in tool_checked if not row['tools_ok']]
    fact_checked = [row for row in results if row['fact_ok'] is not None]
    fact_hits = [row for row in fact_checked if row['fact_ok']]  
    fact_misses = [row for row in fact_checked if not row['fact_ok']]  
    nc_rows = [row for row in results if row['not_covered_ok'] is not None]
    nc_passed = [row for row in nc_rows if row['not_covered_ok']]
    nc_failed = [row for row in nc_rows if not row['not_covered_ok']]

    return {"tool_hits": tool_hits, "tool_misses": tool_misses, "tool_checked":tool_checked, "fact_hits": fact_hits, "fact_misses":fact_misses, "fact_checked":fact_checked, "nc_passed":nc_passed, "nc_failed":nc_failed, "nc_rows":nc_rows}

def summarise_groups(results):
    results = [row for row in results if not row['error']]
    groups = {}
    for row in results:
        groups.setdefault(row['group'],[])
        groups[row['group']].append(row)
    return groups

def save_csv(results,path):
    if not results:
        return 
    fieldnames = list(results[0].keys())
    csv_rows = []
    for row in results:
        csv_rows.append({**row, "tools_used": ", ".join(row['tools_used']), "acceptable_tool_sets": json.dumps(row['acceptable_tool_sets']) , "expected_facts": json.dumps(row['expected_facts'])})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f = f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

if __name__ == "__main__":
    print(f"Loaded {len(rows)} questions")

    # While testing: five rows from different groups. For the full run:  rows_to_run = rows
    test_ids = ["K01", "P02", "C04", "N02", "Z01"]
    test_rows = [row for row in rows if row["id"] in test_ids]

    rows_to_run = rows

    results = []
    for number, row in enumerate(rows_to_run, start=1):
        result = evaluate_row(row)
        results.append(result)
        if result["error"]:
            status = "ERROR"
        else:
            checks = [result["tools_ok"], result["fact_ok"], result["not_covered_ok"]]
            status = "FAIL" if False in checks else "PASS"
        print(f"[{number}/{len(rows_to_run)}] {result['id']}  {result['group']:<12} {status}  {result['tools_used']}")

    save_csv(results, "eval/followup_results.csv")
    print(f"\nSaved {len(results)} rows to eval/followup_results.csv")

    summary = summarise_results(results)
    errors = [row for row in results if row["error"]]

    print("\nTOOL SELECTION")
    print(f"Correct:  {len(summary['tool_hits'])} of {len(summary['tool_checked'])} checked")
    for row in summary["tool_misses"]:
        print(f"  {row['id']}: used {row['tools_used']}, acceptable {row['acceptable_tool_sets']}")

    print("\nFACTS")
    print(f"Correct:  {len(summary['fact_hits'])} of {len(summary['fact_checked'])} checked")
    for row in summary["fact_misses"]:
        print(f"  {row['id']}: expected {row['expected_facts']}")
        print(f"      answer: {row['answer']}")

    print("\nNOT COVERED")
    print(f"Said so:  {len(summary['nc_passed'])} of {len(summary['nc_rows'])}")
    for row in summary["nc_rows"]:
        verdict = "ok" if row["not_covered_ok"] else "DID NOT SAY SO"
        print(f"  {row['id']} [{verdict}]: {row['answer']}")

    print('\nFALSE "NOT COVERED"')
    false_not_covered = [row for row in results
                         if not row["error"]
                         and not row["must_say_not_covered"]
                         and NOT_COVERED_SENTENCE in row["answer"].lower()]
    print(f"Rows:  {len(false_not_covered)}")
    for row in false_not_covered:
        print(f"  {row['id']} (tools used: {row['tools_used']}): {row['answer']}")

    print("\nBY GROUP")
    for group, group_rows in summarise_groups(results).items():
        tools_checked = [row for row in group_rows if row["tools_ok"] is not None]
        tools_passed = [row for row in tools_checked if row["tools_ok"]]
        facts_checked = [row for row in group_rows if row["fact_ok"] is not None]
        facts_passed = [row for row in facts_checked if row["fact_ok"]]
        print(f"{group:<12} {len(group_rows)} rows   tools {len(tools_passed)}/{len(tools_checked)}   "
              f"facts {len(facts_passed)}/{len(facts_checked)}")

    print("\nERRORS")
    print(f"Failed calls:  {len(errors)} of {len(results)}")
    for row in errors:
        print(f"  {row['id']}: {row['error'][:300]}")