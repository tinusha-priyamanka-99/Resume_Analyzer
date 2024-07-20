from langchain_openai import OpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores.pinecone import Pinecone
import os


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)

        docs.append(Document(
            page_content = chunks,
            metadata={"name": filename.name,
                      "id": filename.id,
                      "type":filename.type,
                      "size": filename.size,
                      "unique_id":unique_id},
        ))
    return docs

def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings()
    return embeddings

def push_to_pinecone(pinecone_apikey,
                     pinecone_environment,
                     pinecone_index_name,
                     embeddings,
                     docs):
    pc = pinecone.Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment= pinecone_environment
        )

    index_name= pinecone_index_name
    index = Pinecone.from_documents(docs,embeddings,index_name=index_name)
    

def pull_from_pinecone(pinecone_apikey,
                     pinecone_environment,
                     pinecone_index_name,
                     embeddings):
    pc = pinecone.Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment= pinecone_environment
        )

    index_name= pinecone_index_name
    index = Pinecone.from_existing_index(index_name,embeddings)
    return index

def similar_docs(query,
                 k,
                 pinecone_apikey,
                 pinecone_environment,
                 pinecone_index_name,
                 embeddings,
                 unique_id):
    

    index_name= pinecone_index_name

    index = pull_from_pinecone(pinecone_apikey,
                               pinecone_environment,
                               index_name,
                               embeddings)
    similar_docs = index.similarity_search_with_score(query, 
                                                      int(k),
                                                      {"unique_id":unique_id})
    return similar_docs

def get_summary(current_doc):
    llm = OpenAI(temperature=0)

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary
