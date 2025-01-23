import os

from dotenv import load_dotenv
from groq import Groq
from langchain.vectorstores import Chroma
import gradio as gr

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

# Create a Chroma database
db = Chroma.from_texts(chunks, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setting up the llm
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    # Function for chatbot interaction
    def respond(message, chat_history):
        """
        Gradio function for chatbot interaction.
        Args:
            user_question (str): The user's question.
            history (list): The chat history.
        Returns:
            tuple: Updated chat history and cleared textbox
        """
        # Perform semantic search
        relevant_excerpts = semantic_search(message, retriever)
        # Get the LLM response
        bot_message = resume_chat_completion(
            client, "llama-3.3-70b-versatile", message, relevant_excerpts
        )

        # Append to history and return both history and empty string for textbox
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    # Update the submit method to match the new function signature
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Run the app
if __name__ == "__main__":
    demo.launch()