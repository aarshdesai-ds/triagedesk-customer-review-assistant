from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import load_prompt
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from prompts import analysis_prompt, summary_prompt, positive_prompt, mixed_prompt, followup_prompt
from dotenv import load_dotenv
from schemas import ReviewAnalysis
load_dotenv()

negative_prompt = load_prompt("negative_prompt.json")

model = ChatOpenAI(model = "gpt-4o-mini", temperature= 0)

parser = StrOutputParser()

analysis_chain = analysis_prompt | model.with_structured_output(ReviewAnalysis)

summary_chain = summary_prompt | model | parser

parallel_chain = RunnableParallel({
    "review": RunnableLambda(lambda x: x['review']),
    "analysis": analysis_chain,
    "summary": summary_chain
})


vector_store = Chroma(
    collection_name="kestrel_kb",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="kb_db",
)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3, "filter": {"source": {"$ne": "products.csv"}}}
)

def format_docs(docs):
    pieces= []
    for doc in docs:
        pieces.append(f"[source]: {doc.metadata['source']}\n{doc.page_content}")
    return "\n\n".join(pieces)

def unique_sources(docs):
    sources = []
    for doc in docs:
        if doc.metadata['source'] not in sources:
            sources.append(doc.metadata['source'])
    return sources

def build_query(x):
    issues = x['analysis'].issues
    product = x['analysis'].product
    summary = x['summary']
    if issues:
        return f"{product}: {', ' .join(issues)}. {summary}"
    return "product review loyalty points"

retrieval_step = RunnableParallel({
    "data": RunnablePassthrough(),
    "docs": RunnableLambda(build_query) | retriever
})

def add_context(x):
    data = x['data']
    review = data['review']
    analysis = data['analysis']
    summary = data['summary']
    policy = format_docs(x['docs'])
    sources = unique_sources(x['docs'])
    return {"review":review, "analysis": analysis, "summary": summary, "policy":policy, "sources": sources}

add_context_chain = RunnableLambda(add_context)

positive_chain = positive_prompt | model | parser
negative_chain = negative_prompt | model | parser
mixed_chain = mixed_prompt | model | parser


branch_chain = RunnableBranch(
    (lambda x: x['analysis'].sentiment == "positive", positive_chain),
    (lambda x: x['analysis'].sentiment == "negative", negative_chain),
    (lambda x: x['analysis'].sentiment == "mixed", mixed_chain),
    RunnableLambda(lambda x: "Could not classify the given review")
)

output_chain = RunnableParallel({
    "data": RunnablePassthrough(),
    "reply": branch_chain
}
)

def generate_report(x):

    review = x['data']['review']
    analysis = x['data']['analysis']
    policy = x['data']['policy']
    summary = x['data']['summary']
    sources = x['data']['sources']
    reply = x['reply']

    product = analysis.product
    sentiment = analysis.sentiment
    urgency = analysis.urgency
    issues = analysis.issues 
    
    reviewer = analysis.reviewer
    word_count = len(x['reply'].split())
    escalate = (sentiment == "negative") and (urgency == "high")

    return {
        "review" : review,
        "product": product,
        "sentiment": sentiment,
        "urgency": urgency,
        "issues": issues,
        "reviewer": reviewer,
        "summary": summary,
        "policy": policy,
        "sources": sources,
        "reply": reply,
        "reply_word_count": word_count,
        "escalate": escalate
    }

report_chain = RunnableLambda(generate_report)

final_chain = parallel_chain | retrieval_step | add_context_chain | output_chain | report_chain

build_followup_context = RunnableParallel({
    "question": RunnableLambda(lambda x: x['question']),
    "chat_history": RunnableLambda(lambda x: x['chat_history']),
    "context": RunnableLambda(lambda x: x['product'] + " " + x['question']) | retriever | RunnableLambda(format_docs)
})

followup_chain = build_followup_context | followup_prompt | model | parser
if __name__ == "__main__":
    review = "I ordered the AeroBlend 500 blender three weeks ago. It arrived with a cracked jar and the motor smells like burning plastic after two uses. I want my money back. - Priya S."
    report = final_chain.invoke({"review": review})
    print(followup_chain.invoke({
    "question": "Do you have a store in Pune?",
    "product": "AeroBlend 500 blender",
    "chat_history": [],
}))
