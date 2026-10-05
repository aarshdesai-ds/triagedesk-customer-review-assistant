from langchain_community.document_loaders import TextLoader, DirectoryLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def load_markdown():
    loader = DirectoryLoader(path = "kb" , glob = "*.md", loader_cls= TextLoader, loader_kwargs= {"encoding":"utf-8"})
    docs = loader.load()
    return docs

def load_catalog():
    loader = CSVLoader(file_path="kb/products.csv", encoding="utf-8")
    docs = loader.load()
    return docs

def set_source(docs):
    for doc in docs:
        doc.metadata['source'] = os.path.basename(doc.metadata['source'])
    return docs

def split_markdown(docs):
    splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

def make_ids(docs):
    counters = {}
    ids = []
    for doc in docs:
        source = doc.metadata['source']
        number = counters.get(source, 0)
        ids.append(f"{source}-{number}")
        counters[source] = number + 1
    return ids

def build_store(docs, ids):
    store = Chroma(
        collection_name = "kestrel_kb",
        embedding_function= OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory = "kb_db"
    )
    store.add_documents(docs, ids = ids)
    return store

if __name__ == "__main__":
    markdown_docs = load_markdown()
    markdown_docs = set_source(markdown_docs)
    markdown_chunks = split_markdown(markdown_docs)

    catalog_docs = set_source(load_catalog())

    docs = markdown_chunks + catalog_docs
    ids = make_ids(docs)
    store = build_store(docs, ids)

    print("Markdown chunks:", len(markdown_chunks))
    print("Catalog rows:", len(catalog_docs))
    print("Total in store:", len(store.get()["ids"]))
    print(docs[0].page_content, docs[0].metadata)
    for d in store.similarity_search("opened earbuds return", k=2):
        print(d.metadata["source"], "|", d.page_content[:70])
    
