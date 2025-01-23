import os

from dotenv import load_dotenv
from groq import Groq
from langchain.vectorstores import Chroma

from utils import (
    load_text_data,
    resume_chat_completion,
    semantic_search,
    setup_embedding_model,
)

# Load the embeddings model
embedding_model = setup_embedding_model(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Load the text data
my_resume = load_text_data("resume.txt")

# Chunking the text data
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]
chunk_sizes = [len(chunk) for chunk in chunks]
print("Here are the chunk sizes:", chunk_sizes)

# Create a Chroma database
db = Chroma.from_texts(chunks, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setting up the llm
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# User question
user_question = input("Enter your question: ")

# Semantic search
relevant_excerpts = semantic_search(user_question, retriever)

# LLM Response
response = resume_chat_completion(
    client, "llama-3.3-70b-versatile", user_question, relevant_excerpts
)
print(response)
