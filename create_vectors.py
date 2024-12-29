from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter ,CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
url = [
    "https://www.samsung.com/pk/smartphones/galaxy-a/galaxy-a16-light-green-128gb-sm-a165flggpkd/",
]
doc_web = WebBaseLoader(url).load()


doc = doc_web 

if doc:
    print("Loaded")
else:
    print("Not Loaded")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
split_doc = list(splitter.split_documents(doc))
print(split_doc[:3]) 


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(split_doc, embeddings)
db.save_local("vectors")
print(db.index.ntotal)
try:
    resp = db.add_documents(split_doc)
    print(f"Documents added successfully: {resp}")
except Exception as e:
    print(f"Error adding documents to Chroma DB: {e}")