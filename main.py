from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a polite bot, and your task is helping people as a customer care person"""
        ),
        ("human", "{context}"),
    ]
)

# Initialize embeddings-model 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Initialize vector store
db = FAISS.load_local("vectors", embeddings, allow_dangerous_deserialization=True)


#APIs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


#GROQ
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.3,  # Lower temperature for predictable and precise responses
    max_tokens=50,    # Limit the response length to be concise
    timeout=7,       # Set an appropriate timeout for production-grade use
    max_retries=3,    # Increase retries to handle transient issues
)

chain = prompt_template | llm


def get_bot_response(prompt_template):
    """Get response from the bot."""
    try:
        response = chain.invoke(
            {
                "context": prompt_template,
            }
        )
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Error: {str(e)}"


sentence = "What is the price of samsung A12"
bot_response = get_bot_response(sentence)
print(bot_response)