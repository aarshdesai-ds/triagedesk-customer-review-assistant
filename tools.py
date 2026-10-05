from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import csv

load_dotenv()

store = Chroma(
        collection_name = "kestrel_kb",
        embedding_function= OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory = "kb_db"
    )

retriever = store.as_retriever(
    search_kwargs={"k": 3, "filter": {"source": {"$ne": "products.csv"}}}
)

@tool
def search_policies(query: str) -> str:
    """
    Searches the company's policy documents and returns the most relevant passages with their source file.
    Use this for questions about returns, refunds, shipping, warranty rules, loyalty points, and troubleshooting.
    It does not contain product prices or stock levels.
    Input: a short search phrase, for example "damaged item replacement".
    """
    results = retriever.invoke(query)
    pieces= []
    if not results:
        return "No matching policy was found"
    for doc in results:
        pieces.append(f"[source]: {doc.metadata['source']}\n{doc.page_content}")
    return "\n\n".join(pieces)
   



@tool
def lookup_product(product_name: str) -> str:
    """Returns exact catalog facts for one product: category, price, warranty length in months, and whether it is in stock.
    Use this for any question about a price, a refund amount, how long a warranty lasts, or availability.
    Input: a product name only, for example "AeroBlend 500".
    """

    cleaned_name = product_name.strip().strip("\"'").lower()
    if not cleaned_name:
        return "please give a product name"

    with open("kb/products.csv", newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)

        names = []

        for row in rows:
            if cleaned_name in row["product"].lower() or row['product'].lower() in cleaned_name:
                return f"{row['product']} | category: {row['category']} | price: Rs. {row['price']} | warranty: {row['warranty_months']} months | in stock: {row['in_stock']}"
            names.append(row['product'])
        return f"No product found matching {product_name}. Available products: {', ' .join(names)}" 

@tool
def calculate(expression: str) -> str:
    """
    Works out an arithmetic expression and returns the result.
    Use this for any sum, difference, product or division. Do not do arithmetic in your head.
    Input: numbers and + - * / ( ) only, for example "24 - 20". No words, units or currency symbols.
    """
    cleaned_expression = expression.strip().strip("\"'")
    ALLOWED = "0123456789.+-*/() "
    only_allowed = all(character in ALLOWED for character in cleaned_expression)
    has_power = "**" in cleaned_expression
    if not only_allowed or has_power:
        return "Invalid expression. Use numbers and + - * / ( ) only."
    try:
        result = eval(cleaned_expression, {"__builtins__": {}})
    except Exception:
        return f"Could not calculate '{cleaned_expression}'. Check the expression and try again."
    return str(result)

tools = [search_policies, lookup_product, calculate]

        
if __name__ == "__main__":
    for t in [search_policies, lookup_product]:
        print(f"name:        {t.name}")
        print(f"description: {t.description}")
        print(f"args:        {t.args}")
        print()

    print("--- search_policies ---")
    print(search_policies.invoke("opened earbuds return"))

    print("\n--- lookup_product ---")
    print(lookup_product.invoke("aeroblend"))
    print(lookup_product.invoke("\"AeroBlend 500\""))
    print(lookup_product.invoke("toaster"))
    print(lookup_product.invoke(""))

    print("\n--- calculate ---")
    print(calculate.invoke("24 - 20"))
    print(calculate.invoke("3999 * 0.1"))
    print(calculate.invoke("5 / 0"))
    print(calculate.invoke("import os"))


