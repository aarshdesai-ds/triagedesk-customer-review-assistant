from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from schemas import ReviewAnalysis


analysis_prompt = PromptTemplate(
    template = """
Act as a customer-support analyst. Read the review and pull out the product, sentiment, issues, urgency and reviewer name. Use only what the review actually says. Don't invent issues or a name. 
\n {review} 
""",
input_variables= ['review']
)

summary_prompt = PromptTemplate(
    template = "Please write a one-sentence overview of the following review. It is for an internal agent, so please address it not to the customer but to the system. \n Review: {review}",
    input_variables=['review']
)

positive_prompt = PromptTemplate(
    template="Write an appropriate response to this positive review and the company policy. Mention something specific they praised, and tell them about the loyalty reward described in the policy. Address the reply directly to the customer. Only mention steps that appear in the policy text. Sign off as Customer Support Team. The policy text may contain several help articles. Use only the parts relevant to this review. \n Review: {review} \n Policy: {policy}",
    input_variables=['review', 'policy']
)

negative_prompt = PromptTemplate(
    template="Write an appropriate response to this negative review and the company policy. Apologize for each specific issue the customer described, explain next steps using only the policy, and don't promise anything the policy doesn't say. Address the reply directly to the customer. Only mention steps that appear in the policy text. Sign off as Customer Support Team. The policy text may contain several help articles. Use only the parts relevant to this review. \n Review: {review} \n Policy: {policy}",
    input_variables=['review', 'policy']
)

mixed_prompt = PromptTemplate(
    template="Write an appropriate response to this mixed review and the company policy. Thank them for what they liked, then address the problem using the policy. Address the reply directly to the customer. Only mention steps that appear in the policy text. Sign off as Customer Support Team. The policy text may contain several help articles. Use only the parts relevant to this review.\n Review: {review} \n Policy: {policy}",
    input_variables=['review', 'policy']
)


followup_prompt = ChatPromptTemplate.from_messages([
    ("system","You are an expert support assistant helping an agent with this review. Please reply to the following human question and base your answers on the retrieved context. If the context doesn't contain the answer, say so. Don't guess.  \n Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

react_template = """You are a support assistant helping a human support agent handle one customer review. Answer the agent's questions about this case.

You have access to the following tools:

{tools}

Rules:
1. Base every answer on tool results only. Do not use outside knowledge about the company or its products.
2. For a price, refund amount, warranty length or stock level, use lookup_product.
3. For a question about policy (returns, refunds, shipping, warranty rules, loyalty points, troubleshooting), use search_policies.
4. For any arithmetic, use calculate. Do not work it out yourself.
5. Never say the knowledge base does not cover something unless you have used a tool for this question first. If the tool results do not answer the question, your Final Answer must begin with exactly this sentence: "The knowledge base does not cover this." Then add one short sentence saying what you checked. Do not guess, and do not suggest an answer.6. If no tool is needed, for example when the agent says thank you, skip Action and write "Thought: I now know the final answer" followed by "Final Answer:" and a short, polite reply. The Final Answer must never be empty.
7. Keep the final answer short, and name the source file when you used a policy.
8. When a policy states a condition, check it before answering. Use lookup_product for stock, and compare time limits against what the review says. State clearly whether each condition is met.
9. Report amounts exactly as the tools give them. Do not add or change a currency.
10. If a fact needed to check a condition is not stated in the review or a tool result, such as the delivery date, do not assume it. Say what the agent needs to confirm with the customer.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Product: {product}
Review: {review}

Conversation so far:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(react_template)

if __name__ == "__main__":
    negative_prompt.save("negative_prompt.json")
    print(react_prompt.input_variables)
    print(react_prompt.format(
        tools="(tool descriptions go here)",
        tool_names="search_policies, lookup_product, calculate",
        product="AeroBlend 500 blender",
        review="It arrived with a cracked jar. I want my money back. - Priya S.",
        chat_history="(none yet)",
        input="How much would her refund be?",
        agent_scratchpad="",
    ))