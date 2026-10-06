from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from tools import tools

load_dotenv()

FALLBACK_STRING = "I couldn't work that out within the step limit. Please try rephrasing the question."

model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=2048)

model_with_tools = model.bind_tools(tools)

MAX_ROUNDS = 5

tools_by_name = {t.name: t for t in tools}

def text_of(message):
    """Return the reply as plain text, whether the model gives a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )

SYSTEM_RULES = """You are a support assistant helping a human support agent handle one customer review. Answer the agent's questions about this case.

Rules:
1. Base every answer on tool results only. Do not use outside knowledge about the company or its products.
2. For a price, refund amount, warranty length or stock level, use lookup_product.
3. For a question about policy (returns, refunds, shipping, warranty rules, loyalty points, troubleshooting), use search_policies.
4. For any arithmetic, use calculate. Do not work it out yourself.
5. Never say the knowledge base does not cover something unless you have used a tool for this question first. If the tool results do not answer the question, your answer must begin with exactly this sentence: "The knowledge base does not cover this." Then add one short sentence saying what you checked. Do not guess, and do not suggest an answer.
6. If no tool is needed, for example when the agent says thank you, reply briefly without calling one.
7. Keep the answer short, and name the source file when you used a policy.
8. When a policy states a condition, check it before answering. Use lookup_product for stock, and compare time limits against what the review says. State clearly whether each condition is met.
9. Report amounts exactly as the tools give them. Do not add or change a currency.
10. If a fact needed to check a condition is not stated in the review or a tool result, such as the delivery date, do not assume it. Say what the agent needs to confirm with the customer.

The case:"""

NOT_COVERED_SENTENCE = "the knowledge base does not cover this"

NUDGE_MESSAGE = ("Your previous answer is withdrawn, because you had not used any tool for this question. "
                 "Use search_policies or lookup_product now. "
                 "If the tool result answers the question, answer it directly and do not say the knowledge base does not cover it. "
                 "Only use that sentence if the tool result truly does not answer the question.")

def to_messages(history):
    messages = []
    for question, answer in history:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages

def run_agent(messages):
    tools_used = []
    nudged = False
    for i in range(MAX_ROUNDS):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            answer = text_of(response)
            claims_not_covered = NOT_COVERED_SENTENCE in answer.lower()
            if claims_not_covered and not tools_used and not nudged:
                messages.append(HumanMessage(content=NUDGE_MESSAGE))
                nudged = True
                continue
            return answer, tools_used

        for tool_call in response.tool_calls:
            tools_used.append(tool_call['name'])
            tool = tools_by_name.get(tool_call['name'])
            if not tool:
                messages.append(ToolMessage(content = "Unknown tool call", tool_call_id = tool_call['id']))
                continue
            messages.append(tool.invoke(tool_call))
    return FALLBACK_STRING, tools_used

def ask_detailed(question, product, review, history):
    messages = [SystemMessage(content = f'{SYSTEM_RULES}\nProduct:{product}\nReview:{review}')] + to_messages(history) + [HumanMessage(question)]
    answer, tools_used = run_agent(messages)
    return {"answer":answer, "tools_used":tools_used}

def ask(question, product, review, history):
     return ask_detailed(question, product, review, history)['answer']

if __name__ == "__main__":
    product = "AeroBlend 500 blender"
    review = ("I ordered the AeroBlend 500 blender three weeks ago. It arrived with a cracked jar "
              "and the motor smells like burning plastic after two uses. I want my money back. - Priya S.")
    print(ask_detailed("How much would her refund be?", product, review, []))
    questions = [
        "How much would her refund be?",
        "If she had bought it 20 months ago, would it still be under warranty?",
        "She'd take a replacement. Can we send one?",
        "Do you have a store in Pune?",
        "Thanks, that's all.",
    ]

    history = []
    for number, question in enumerate(questions, start=1):
        print(f"\n{'=' * 70}")
        print(f"QUESTION {number}: {question}")
        print("=" * 70)
        answer = ask(question, product, review, history)
        print(f"\nANSWER {number}: {answer}")
        history.append((question, answer))