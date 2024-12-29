from azure.cosmos import CosmosClient, exceptions
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# Configurations
PDF_PATH = "data/Certificate-Form-Final.pdf"
DB_NAME = "BudgetDB"
CONTAINER_NAME = "BudgetContainer"
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




def load_and_split_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


def generate_embeddings(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        api_key=api_key,
        model="models/embedding-001"  # Replace with the appropriate model
    )
    return embeddings


def save_to_cosmos(vectors, chunks, endpoint, key, db_name, container_name):
    client = CosmosClient(endpoint, key)
    database = client.create_database_if_not_exists(db_name)
    container = database.create_container_if_not_exists(id=container_name, partition_key='/id')

    for idx, (vector, chunk) in enumerate(zip(vectors, chunks)):
        item = {
            "id": str(idx),  # Unique ID for each document
            "content": chunk.page_content,  # Actual content of the chunk
            "metadata": chunk.metadata,  # Metadata related to the chunk
            "vector": vector,  # Ensure the vector
        }
        container.upsert_item(item)



def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


def fetch_closest_documents(query_vector, endpoint, key, db_name, container_name):
    client = CosmosClient(endpoint, key)
    database = client.get_database_client(db_name)
    container = database.get_container_client(container_name)

    items = list(container.read_all_items())
    similarities = [
        (cosine_similarity(query_vector, item['vector']), item) for item in items
    ]
    sorted_items = sorted(similarities, key=lambda x: x[0], reverse=True)
    return [item[1] for item in sorted_items[:5]]  # Top 5 matches


def generate_response(query, context, api_key):
    model = GoogleGenerativeAI(model="gemini-pro", api_key=api_key)
    prompt = f"Based on the following context:\n{context}\nAnswer the following query: {query}"
    print("model initialized")
    return model(prompt)


def chat_with_budget_bot(query, embeddings_api_key, cosmos_endpoint, cosmos_key, db_name, container_name, gemini_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=embeddings_api_key)
    query_vector = embeddings.embed_query(query)

    closest_docs = fetch_closest_documents(query_vector, cosmos_endpoint, cosmos_key, db_name, container_name)
    context = "\n".join(doc["content"] for doc in closest_docs)

    response = generate_response(query, context, gemini_api_key)

    return response


# Process and Save Data
try:
    chunks = load_and_split_document(PDF_PATH)
    embeddings = generate_embeddings(chunks, GOOGLE_API_KEY)
    save_to_cosmos(embeddings, chunks, COSMOS_ENDPOINT, COSMOS_KEY, DB_NAME, CONTAINER_NAME)
    print("Data store created successfully!")
except Exception as e:
    print(f"Exception: {e}")
