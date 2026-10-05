# TriageDesk: A Customer-Review Assistant with Measured Improvements

A LangChain project that triages customer reviews for a fictional online store. A fixed pipeline analyses each review, retrieves the relevant policy, drafts a reply and flags urgent complaints for a supervisor. A tool-calling agent then answers the support agent's follow-up questions. Both parts have a labelled evaluation set, and every change to the system was measured against it.

*This is an independent learning project. Kestrel is a fictional store. All reviews, policies and products are synthetic and were written for this project.*

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Knowledge Base](#knowledge-base)
- [Evaluation: the Pipeline](#evaluation-the-pipeline)
- [Evaluation: the Agent](#evaluation-the-agent)
- [What the Evaluations Taught](#what-the-evaluations-taught)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Known Limitations](#known-limitations)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

A support agent pastes in a customer review. The system:

1. **Analyses** it: product, sentiment, issues, urgency and the reviewer's name, as structured output.
2. **Retrieves** the relevant store policy from a vector store.
3. **Drafts a reply** with a prompt chosen by sentiment (positive, negative or mixed), using only the retrieved policy.
4. **Flags** negative, high-urgency reviews for a supervisor.
5. **Answers follow-up questions** through an agent that can search the policies, look up exact catalog facts and do arithmetic.

The second half of the project is the evaluation. Two labelled test sets and two scripts measure the pipeline and the agent, and the sections below record what each change did to the scores, including the changes that made things worse.

---

## How It Works

### The pipeline (`chains.py`)

```
 review
    │
    ├─► analysis  (product, sentiment, issues, urgency, reviewer)
    └─► summary   (one sentence for the support agent)
    │
    ▼
 build a search query from the product, the issues and the summary
    │
    ▼
 retrieve 3 policy chunks  (catalog rows are filtered out)
    │
    ▼
 draft a reply:  positive prompt / negative prompt / mixed prompt
    │
    ▼
 report:  analysis, sources, reply, and  escalate = negative sentiment AND high urgency
```

The pipeline is a fixed chain. Its steps are known in advance, so nothing in it is left to an agent.

### The follow-up agent (`tools.py`, `agent.py`)

| Tool | Input | Returns |
|---|---|---|
| `search_policies` | a search phrase | the nearest policy passages, each with its source file |
| `lookup_product` | a product name | the exact catalog row: category, price, warranty months, stock |
| `calculate` | an arithmetic expression | the result; the input is checked character by character before it is evaluated |

The agent is a hand-written tool-calling loop built on `bind_tools`: invoke the model, run every tool call it asks for, return each result as a tool message, and stop after five rounds.

The loop also enforces one rule in code. If the model says the knowledge base does not cover a question without having called any tool, the answer is rejected once and the model is told to use a tool first. The reason for this is in [the agent evaluation](#evaluation-the-agent).

`agent_react.py` holds the earlier version of the agent, a text-based ReAct agent built with `create_react_agent` and `AgentExecutor`. It is kept so the two can be compared on the same questions.

---

## Knowledge Base

Everything is synthetic. `ingest.py` loads it into one Chroma collection of 21 chunks.

| Source | Content | Chunks |
|---|---|---|
| `returns.md` | Return window, condition, damaged or defective items, refund timing, return shipping | 5 |
| `shipping.md` | Delivery times, delivery costs, late deliveries, tracking | 3 |
| `troubleshooting.md` | Bluetooth drops, a blender's burning smell, charging | 3 |
| `warranty.md` | Coverage, exclusions, claims, extended warranty | 2 |
| `loyalty.md` | Earning and redeeming points, expiry and tiers | 2 |
| `products.csv` | Six products with category, price, warranty months and stock | 6 |

Markdown files are split with a Markdown-aware splitter (300 characters, 50 overlap). Each catalog row is one chunk. Chunk IDs are fixed, so re-running `ingest.py` does not create duplicates.

---

## Evaluation: the Pipeline

`evaluate.py` runs the pipeline over the labelled reviews in `eval/reviews.jsonl` and saves per-review results to `eval/results.csv`.

```bash
python evaluate.py
```

Each review is labelled with the acceptable sentiments, the product, whether it should be escalated, the policy files that should be retrieved, and the reviewer's name (or none, to test whether a name is invented).

| Group | Reviews | What it tests |
|---|---|---|
| `urgent_complaint` | 8 | Safety hazards, withheld money, threats: must be escalated |
| `routine_complaint` | 7 | Ordinary complaints: must not be escalated |
| `mixed` | 7 | Praise and a problem in one review |
| `positive` | 7 | Praise only |
| `policy_specific` | 6 | Each depends on one particular rule, such as the 14-day limit on opened earbuds |
| `tricky` | 8 | Sarcasm, no product named, a product not in the catalog, a one-word review, text-speak |
| `fresh_*` | 11 | Added later as a held-out test; see below |

### Four runs, one change at a time

These runs used the first 43 reviews.

| Run | Change | Urgent caught | False escalations | Sentiment | Source retrieved |
|---|---|---|---|---|---|
| Baseline | Default model and temperature | 7 of 8 | 1 of 29 | 41 of 43 | 32 of 41 |
| 1 | `gpt-4o-mini`, temperature 0 | 8 of 8 | 4 of 29 | 43 of 43 | 36 of 41 |
| 2 | Catalog rows filtered out of the search | 8 of 8 | 4 of 29 | 43 of 43 | 39 of 41 |
| 3 | Urgency levels defined as a rubric | 8 of 8 | 0 of 29 | 42 of 43 | 39 of 41 |

Product extraction (40 of 40) and reviewer extraction (43 of 43, with no invented names) were correct in every run.

What each run showed:

- **Baseline.** The same review was escalated in a five-review trial and missed in the full run, with no change between them. The model was running at its default temperature, so its urgency ratings varied from run to run.
- **Run 1.** Naming the model and setting temperature 0 made the results repeatable. The new model caught every urgent review and also over-rated four ordinary ones.
- **Run 2.** Catalog rows appeared in the sources of 42 of 43 reviews, taking one of three retrieval slots. One filter fixed three retrieval misses.
- **Run 3.** The schema's description of `urgency` did not define its levels. A rubric removed all four false escalations and kept every urgent review. It also changed one sentiment label, which shows that editing one field's description can move another.

Per-review results for each run are in `eval/results_baseline.csv` and `eval/results_step1.csv` to `eval/results_step3.csv`.

### The held-out test

A score of 8 of 8 and 0 of 29 came after the rubric was written with the failing reviews in view. To test it fairly, 11 new reviews were written to avoid every word in the rubric and to set tone against severity.

| | First 43 reviews | 11 held-out reviews |
|---|---|---|
| Urgent caught | 8 of 8 | **3 of 5** |
| False escalations | 0 of 29 | 0 of 4 |

The two misses have the same cause: **a serious event described calmly is under-rated.**

- A customer charged twice for a monitor, writing politely, was rated mixed sentiment and medium urgency.
- An electric shock from a blender, mentioned inside a happy review that ends "it has been fine since", was rated mixed and medium.

The rubric's rule about tone works in one direction only. Angry wording no longer raises the urgency, but polite or reassuring wording still lowers it. The second case also exposes a flaw in the escalation rule itself: it requires negative sentiment, so a hazard reported by a satisfied customer can never be escalated.

The 11 reviews have since been added to the main set. On all 54 reviews, the current system catches 11 of 13 urgent reviews with 0 of 33 false escalations, 51 of 54 on sentiment, and 39 of 41 on retrieval. These gaps are open; see [Known Limitations](#known-limitations).

---

## Evaluation: the Agent

`evaluate_agent.py` asks the agent 24 labelled questions from `eval/followup.jsonl` and saves the results to `eval/followup_results.csv`.

```bash
python evaluate_agent.py
```

| Group | Questions | What it tests |
|---|---|---|
| `catalog` | 5 | Prices, warranty months and stock |
| `policy` | 7 | Return windows, refund timing, fees, loyalty points, delivery area |
| `calculate` | 4 | Arithmetic on looked-up values |
| `combined` | 3 | A catalog fact and a policy in one question |
| `not_covered` | 3 | Questions the knowledge base cannot answer |
| `no_tool` | 2 | A thank-you |

Three things are checked:

- **Tool selection.** The set of tools used must match an acceptable set.
- **Facts.** The answer must contain the expected value, such as "3999", "5-7" or "1899.9". This store's answers are exact, so a correct answer can be checked by its content and not only by the tools behind it.
- **Not covered.** For a question with no answer in the knowledge base, the agent must say so with a fixed sentence, and must have used a tool first.

### From ReAct to tool calling

| Stage | Tool selection | Facts | "Not covered", after checking | Calculator used when needed |
|---|---|---|---|---|
| ReAct agent, first run | 18 of 21 | 17 of 19 | not measured strictly | 2 of 4 |
| ReAct agent, with a fixed "not covered" sentence | 19 of 21 | 18 of 19 | 1 of 3 | 2 of 4 |
| Tool-calling agent, same rules | 20 of 21 | 18 of 19 | 0 of 3 | 4 of 4 |
| Tool-calling agent, with the guardrail in the loop | **21 of 21** | **19 of 19** | **3 of 3** | 4 of 4 |

The ReAct agent followed its rules inconsistently. It skipped the calculator on two of four questions and gave a wrong amount on one of them (Rs. 1899 for an extended warranty that costs Rs. 1899.90). When a rule required an exact opening sentence, one question failed to parse five times in a row and hit the step limit. The tool-calling agent, given the same rules word for word, used the calculator every time and had no parsing failures.

Per-question results are in `eval/followup_results_baseline.csv` (ReAct, first run), `eval/followup_results_toolcalling.csv`, `eval/followup_results_honest_checker.csv` and `eval/followup_results_guardrail.csv`. The per-question file for the second ReAct run was not kept.

### The score that hid a failure

After the move to tool calling, the "not covered" check reported 3 of 3. Reading the rows showed that in all three, and in a fourth question, the agent had called no tool at all and then written "I checked":

```
 tools used: []
 "The knowledge base does not cover this. I checked for shipping policies."
```

The check had looked only for the sentence, so it rewarded an answer that skipped the work. One of the four was also wrong: the shipping policy does answer that question.

Two changes followed. The check now requires a tool call as well as the sentence, which took the honest score to 0 of 3. Then the rule was moved out of the prompt and into the loop, which rejects that answer once when no tool has been used. Two prompts had asked for this and been ignored. With the rule in code, all three questions are checked before they are answered.

---

## What the Evaluations Taught

- **Fix the instrument first.** With the model at its default temperature, one review flipped between runs. No change could be measured until the output was repeatable.
- **One change per run.** Each row in the tables differs from the one above by a single change, so each gain has a known cause.
- **A perfect score on a set you tuned against proves little.** The rubric scored 8 of 8 on the reviews it was shaped around and 3 of 5 on reviews it had not seen.
- **A metric can be met in a way that defeats it.** The agent learned to produce the required sentence without doing the check behind it. Only reading the rows showed this.
- **When a rule must hold, put it in code.** A prompt asks. A loop can refuse.
- **String checks confirm that a fact is present, not that an answer makes sense.** One answer passes the fact check while contradicting itself; see below.

---

## Project Structure

```
triagedesk-customer-review-assistant/
│
├── README.md
├── requirements.txt
├── .env                    # Not committed: OPENAI_API_KEY
│
├── kb/                     # Five policy files and the product catalog
├── kb_db/                  # Not committed: the Chroma store, built by ingest.py
│
├── ingest.py               # Load, split, embed and store the knowledge base
├── schemas.py              # ReviewAnalysis, including the urgency rubric
├── prompts.py              # Analysis, summary and reply prompts; the ReAct prompt
├── negative_prompt.json    # The negative-reply prompt, saved and loaded as JSON
├── chains.py               # The triage pipeline
├── tools.py                # The agent's three tools
├── agent.py                # The tool-calling agent, with the guardrail
├── agent_react.py          # The earlier ReAct agent, kept for comparison
├── main.py                 # Command-line report and follow-up chat
│
├── evaluate.py             # Pipeline evaluation
├── evaluate_agent.py       # Agent evaluation
└── eval/
    ├── reviews.jsonl               # 54 labelled reviews
    ├── fresh_reviews.jsonl         # The 11 held-out reviews, before they were merged in
    ├── followup.jsonl              # 24 labelled agent questions
    └── *results*.csv               # Per-row results for every run described above
```

---

## How to Run

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/aarshdesai-ds/triagedesk-customer-review-assistant.git
   cd triagedesk-customer-review-assistant
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project folder:
   ```
   OPENAI_API_KEY=your-key-here
   ```

4. Build the knowledge base (one embedding call):
   ```bash
   python ingest.py
   ```

5. Run the report and follow-up chat:
   ```bash
   python main.py
   ```

6. Run the evaluations. These call the model: about 160 calls for the pipeline and about 70 for the agent.
   ```bash
   python evaluate.py
   python evaluate_agent.py
   ```

The project uses `gpt-4o-mini` at temperature 0 and `text-embedding-3-small`. `tools.py` and `agent.py` can also be run on their own to see their checks.

---

## Known Limitations

- **Serious events described calmly are under-rated.** Two of five held-out urgent reviews were missed for this reason.
- **Escalation requires negative sentiment.** A hazard mentioned in a positive or mixed review is never escalated, whatever urgency it is given.
- **One agent answer contradicts itself.** Asked about delivery to another country, the agent opens with "The knowledge base does not cover this" and then quotes the policy that answers the question. The fact check passes because the right word is present. The guardrail's rejected first answer stays in the conversation, which is the likely cause.
- **Two retrieval misses remain.** A fan remote that "stopped working" retrieves troubleshooting text when the reply needs the warranty policy, and a product that is not in the catalog retrieves the loyalty policy, most likely on the shared brand name.
- **The search query falls back to the loyalty policy** whenever a review has no listed issues, whatever its sentiment. The current test set no longer triggers this, so its effect is unmeasured.
- **The test sets are small and synthetic,** and most of the labels and the rubric were written by the same hand. The agent's 24 questions have not been tested against a held-out set, and the pipeline's held-out result suggests one would find gaps.
- **Each stage was measured once.** One agent answer changed between runs with no change aimed at it.
- **Replies are not checked against the policy.** The evaluation confirms that the right policy was retrieved, not that the reply applied it correctly.
- **The "not covered" guardrail sends one reminder.** If the model ignores it, the unchecked answer goes through.
- **The knowledge base is tiny and fictional.** Six products and five short policy files.

---

## Ideas for Extension

1. **Escalate on high urgency whatever the sentiment,** and measure the effect on both review sets.
2. **A rule layer for escalation.** Fixed phrases for hazards and payment errors, combined with the model's rating by "or", so that a calm description cannot lower the result.
3. **A grounding check for replies.** A second chain with structured output that lists any claim in a reply the retrieved policy does not support, checked against a hand-read sample before its numbers are trusted.
4. **A held-out question set for the agent,** with mixed questions, products outside the catalog and calculations phrased in words.
5. **Repeat each run several times** to measure how much the scores vary without any change.
6. **Retrieval experiments,** one at a time: a fourth chunk per search, chunk headers, and a larger chunk size.
