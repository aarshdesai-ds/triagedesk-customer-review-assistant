# TriageDesk: A Customer-Review Assistant with Measured Improvements

A LangChain project that triages customer reviews for a fictional online store. A fixed pipeline analyses each review, retrieves the relevant policy, drafts a reply and flags urgent complaints for a supervisor. A tool-calling agent then answers the support agent's follow-up questions. Both parts have a labelled evaluation set, and every change to the system was measured against it, including a change of model from `gpt-4o-mini` to Claude Haiku 4.5.

*This is an independent learning project. Kestrel is a fictional store. All reviews, policies and products are synthetic and were written for this project.*

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Knowledge Base](#knowledge-base)
- [Evaluation: the Pipeline](#evaluation-the-pipeline)
- [Evaluation: the Agent](#evaluation-the-agent)
- [Model Comparison: gpt-4o-mini and Claude Haiku 4.5](#model-comparison-gpt-4o-mini-and-claude-haiku-45)
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

The project was built and tuned on `gpt-4o-mini`. It was then run on Claude Haiku 4.5 with the prompts, rubric, tools and test sets unchanged. The pipeline scored higher on Claude. The agent scored lower, and the reasons are set out in [the model comparison](#model-comparison-gpt-4o-mini-and-claude-haiku-45).

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

### The model

The chat model is created in one place in `chains.py` and one place in `agent.py`. Everything else (retrieval, prompts, structured output, tools, the loop) goes through LangChain's interfaces and does not depend on the provider. Embeddings use `text-embedding-3-small` for both models, so both search the same Chroma index.

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

These runs used `gpt-4o-mini` and the first 43 reviews.

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

The 11 reviews have since been added to the main set. On all 54 reviews, `gpt-4o-mini` catches 11 of 13 urgent reviews with 0 of 33 false escalations, 51 of 54 on sentiment, and 39 of 41 on retrieval. The same 54 reviews were later run on Claude Haiku 4.5; see [the model comparison](#model-comparison-gpt-4o-mini-and-claude-haiku-45).

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

These runs used `gpt-4o-mini`.

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

## Model Comparison: gpt-4o-mini and Claude Haiku 4.5

The whole system was tuned on `gpt-4o-mini`. To see how much of that tuning carried over, both evaluations were run again on Claude Haiku 4.5 (`claude-haiku-4-5-20251001`, temperature 0, `max_tokens` 2048).

Nothing else was changed: the same prompts, urgency rubric, tools, guardrail, Chroma index and test sets. The comparison is between a system tuned on one model and the same system moved to another without tuning, so it measures how well the design transfers and not which model is better.

### What changed in the code

| Change | Where | Why |
|---|---|---|
| `ChatOpenAI` replaced with `ChatAnthropic` | `chains.py`, `agent.py` | The model swap itself |
| `langchain-anthropic` added | `requirements.txt` | The Claude model class |
| A `text_of()` helper that reads the reply as plain text | `agent.py` | See below |

The swap was not quite one line. The first agent run on Claude crashed on one question:

```
C01: 'list' object has no attribute 'lower'
```

The loop checked the reply with `response.content.lower()`. `gpt-4o-mini` always returned the reply as a string. Claude can return a list of content blocks, and on this question it returned an empty one. `text_of()` handles both shapes. With that fix, all 24 questions ran without errors, and the results below are from that run.

### The pipeline: higher on Claude

All 54 reviews, one run of each model.

| Measure | `gpt-4o-mini` | Claude Haiku 4.5 |
|---|---|---|
| Urgent reviews caught | 11 of 13 | **12 of 13** |
| False escalations | 0 of 33 | 0 of 33 |
| Sentiment | 51 of 54 | **54 of 54** |
| Policy source retrieved | 39 of 41 | **40 of 41** |
| Product | not recorded on 54 | 51 of 51 |
| Reviewer | not recorded on 54 | 54 of 54 |
| Failed calls | 0 | 0 |

On the 11 held-out reviews, Claude caught 4 of 5 urgent reviews, against 3 of 5.

The one review Claude missed is the most useful row in the table. It is the blender electric shock inside a happy review:

```
MISSED F05 (sentiment mixed, urgency high)
```

`gpt-4o-mini` rated this review mixed sentiment and **medium** urgency, so it failed for two reasons. Claude rated it mixed sentiment and **high** urgency. The urgency is now right, and the review is still missed, because the escalation rule requires negative sentiment. Changing the model separated a fault in the model's judgment from a fault in the pipeline's rule. The first went away. The second, which was already listed under Known Limitations, is now the only cause.

The other held-out miss, the politely reported double charge, was rated high urgency and escalated.

The remaining retrieval miss (M05) is unchanged: a fan remote that "stopped working" retrieves troubleshooting text when the reply needs the warranty policy.

### The agent: lower on Claude

All 24 questions, one run of each model. The `gpt-4o-mini` column is the final tool-calling agent with the guardrail.

| Measure | `gpt-4o-mini` | Claude Haiku 4.5 |
|---|---|---|
| Tool selection | 21 of 21 | 20 of 21 |
| Facts | 19 of 19 | 16 of 19 |
| "Not covered", after checking | 3 of 3 | 3 of 3 |
| False "not covered" | 0 | 2 |
| Calculator used when needed | 4 of 4 | 3 of 4 |
| Failed calls | 0 | 0 |

Three questions failed, each in a different way.

**C02: one tool, then "not covered".** The question needs a product price, the extended-warranty rule from the policies, and a calculation. Claude looked up the product, found no warranty pricing in the catalog row, and stopped:

```
tools used: ['lookup_product']
"The knowledge base does not cover this. I checked the product catalog, which shows
the standard warranty is 36 months, but does not include pricing for extended warranties."
```

The guardrail did not fire, because it only requires that some tool was used. It was written for the failure `gpt-4o-mini` showed, which was claiming to have checked while calling no tool at all. Claude does call a tool, and then concludes too early. The answer is honest about what was checked and wrong about what the knowledge base contains.

**P07: a false "not covered" after searching.** Asked about delivery to another country, Claude searched the policies and still reported that the knowledge base does not cover it. The expected answer names India. Whether the right passage was retrieved for Claude's search phrase has not been checked yet, so this could be a retrieval miss or a reasoning miss.

On `gpt-4o-mini`, this same question produced an answer that opened with "The knowledge base does not cover this" and then quoted the policy that answers it. That answer passed the fact check because the right word was present. Claude's answer is consistent and fails the check. Neither is correct, and the earlier pass was the weaker result.

**C01: the right tools, then an empty answer.** Claude called `lookup_product` and `calculate`, which is the correct pair, and then returned a final reply with no text. The loop treats any reply without tool calls as the final answer, so it returned an empty string. This is the same reply that crashed the first run. The cause has not been confirmed.

### What the comparison showed

- **The fixed pipeline transferred well.** Structured output, a rubric and a retrieval filter carried over to a different model with no changes and scored higher.
- **The agent's guardrail did not.** It was shaped around one model's failure and did not catch a different model's failure. The rule needs to describe what a good answer requires, not what one model happened to do wrong.
- **A model swap is a change to validate.** The code change was small. Finding out what it did took two evaluation runs and reading the rows.
- **The second model exposed a weak check.** P07 passed on `gpt-4o-mini` for the wrong reason, and only the comparison made that obvious.

Each model was run once. Per-row results for the Claude runs are in `eval/results_claude.csv` and `eval/followup_results_claude_baseline.csv`.

---

## What the Evaluations Taught

- **Fix the instrument first.** With the model at its default temperature, one review flipped between runs. No change could be measured until the output was repeatable.
- **One change per run.** Each row in the tables differs from the one above by a single change, so each gain has a known cause.
- **A perfect score on a set you tuned against proves little.** The rubric scored 8 of 8 on the reviews it was shaped around and 3 of 5 on reviews it had not seen.
- **A metric can be met in a way that defeats it.** The agent learned to produce the required sentence without doing the check behind it. Only reading the rows showed this.
- **When a rule must hold, put it in code.** A prompt asks. A loop can refuse.
- **A rule written against one model's failure may not hold for another.** The guardrail stopped `gpt-4o-mini` from skipping the tools. Claude used a tool and still concluded too early, and the guardrail let it through.
- **Changing the model can tell you where a fault lives.** A review that failed for two reasons on one model failed for one reason on the other, which pointed at the escalation rule and away from the model.
- **String checks confirm that a fact is present, not that an answer makes sense.** One `gpt-4o-mini` answer passed the fact check while contradicting itself.

---

## Project Structure

```
triagedesk-customer-review-assistant/
│
├── README.md
├── requirements.txt
├── .env                    # Not committed: ANTHROPIC_API_KEY and OPENAI_API_KEY
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
    └── *results*.csv               # Per-row results for every run described above, for both models
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
   python -m pip install -r requirements.txt
```

3. Create a `.env` file in the project folder. The chat model uses the Anthropic key. The embeddings use the OpenAI key.

```
   ANTHROPIC_API_KEY=your-key-here
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

6. Run the evaluations. These call the model: about 160 calls for the pipeline and about 70 for the agent. Each run overwrites `eval/results.csv` or `eval/followup_results.csv`, so rename the file afterwards if you want to keep it.

```bash
   python evaluate.py
   python evaluate_agent.py
```

The project currently runs on `claude-haiku-4-5-20251001` at temperature 0, with `text-embedding-3-small` for embeddings. The earlier results in this README were produced with `gpt-4o-mini` at temperature 0. To reproduce them, replace `ChatAnthropic(...)` with `ChatOpenAI(model="gpt-4o-mini", temperature=0)` where the model is created in `chains.py` and `agent.py`.

`tools.py` and `agent.py` can also be run on their own to see their checks.

---

## Known Limitations

- **Escalation requires negative sentiment.** A hazard mentioned in a positive or mixed review is never escalated, whatever urgency it is given. On Claude this is the only remaining cause of a missed escalation.
- **Serious events described calmly are under-rated by `gpt-4o-mini`.** Two of five held-out urgent reviews were missed for this reason. Claude rated both as high urgency, in one run.
- **The "not covered" guardrail only checks that some tool was used.** One catalog lookup is enough to pass it, even when the answer is in the policies. Claude gave one false "not covered" answer this way.
- **An empty final answer is returned as it is.** On one question Claude used the right tools and then replied with no text, and the loop returned an empty string. The cause has not been confirmed.
- **One policy question is answered wrongly by both models.** Asked about delivery to another country, `gpt-4o-mini` contradicts itself and passes the fact check, and Claude says the knowledge base does not cover it.
- **One retrieval miss remains on both models.** A fan remote that "stopped working" retrieves troubleshooting text when the reply needs the warranty policy. On `gpt-4o-mini`, a product that is not in the catalog also retrieves the loyalty policy.
- **The search query falls back to the loyalty policy** whenever a review has no listed issues, whatever its sentiment. The current test set no longer triggers this, so its effect is unmeasured.
- **The system was tuned on one model and only measured on the other.** No prompt, rubric or guardrail was adjusted for Claude, so its agent scores are a starting point and not its best.
- **The test sets are small and synthetic,** and most of the labels and the rubric were written by the same hand. The agent's 24 questions have not been tested against a held-out set, and the pipeline's held-out result suggests one would find gaps.
- **Each stage was measured once, on each model.** One agent answer changed between `gpt-4o-mini` runs with no change aimed at it, so small differences between the two models may not be stable.
- **Replies are not checked against the policy.** The evaluation confirms that the right policy was retrieved, not that the reply applied it correctly.
- **The "not covered" guardrail sends one reminder.** If the model ignores it, the unchecked answer goes through.
- **The knowledge base is tiny and fictional.** Six products and five short policy files.

---

## Ideas for Extension

1. **Handle an empty final answer in the loop.** Ask once for a final answer when the reply has no text, and measure the effect on both models.
2. **Require a policy search before "not covered".** Reject that answer unless `search_policies` was called, and confirm it fixes the early stop on Claude without lowering the `gpt-4o-mini` score.
3. **Find the cause of the delivery-area miss.** Read what `search_policies` returned for each model's search phrase, to tell a retrieval miss from a reasoning miss.
4. **Escalate on high urgency whatever the sentiment,** and measure the effect on both review sets. One mixed, high-urgency review is labelled as not needing escalation, so this may trade a miss for a false alarm.
5. **A rule layer for escalation.** Fixed phrases for hazards and payment errors, combined with the model's rating by "or", so that a calm description cannot lower the result.
6. **A grounding check for replies.** A second chain with structured output that lists any claim in a reply the retrieved policy does not support, checked against a hand-read sample before its numbers are trusted.
7. **A held-out question set for the agent,** with mixed questions, products outside the catalog and calculations phrased in words.
8. **Repeat each run several times on each model** to measure how much the scores vary without any change.
9. **Retrieval experiments,** one at a time: a fourth chunk per search, chunk headers, and a larger chunk size.