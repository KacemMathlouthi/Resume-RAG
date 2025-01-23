import os

import gradio as gr
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
my_resume = load_text_data("data/resume.txt")

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
    gr.Markdown("# Kacem Mathlouthi Resume RAG Chatbot")
    gr.Markdown("""
    ## About this Chatbot
    
    This is a Retrieval-Augmented Generation (RAG) chatbot powered by AI that allows you to interactively explore Kacem Mathlouthi's resume. 
    
    - **Technology**: Utilizes advanced semantic search and language model techniques
    - **Purpose**: Provide detailed, context-aware answers about Kacem's professional background
    - **How it works**: 
      1. Your question is semantically searched against resume chunks
      2. Relevant excerpts are retrieved
      3. A language model generates a precise, contextual response
    """)

    chatbot = gr.Chatbot(type="messages", height=400)
    with gr.Row(equal_height=True):
        with gr.Column(scale=10):
            msg = gr.Textbox(label="Ask a question about the resume", container=False)
        with gr.Column(scale=1):
            submit = gr.Button(value="➤", size="sm")
    
    clear = gr.ClearButton([msg, chatbot], size="sm")

    # Function for chatbot interaction
    def respond(message, chat_history):
        """
        Gradio function for chatbot interaction.
        Args:
            message (str): The user's question.
            chat_history (list): The chat history.
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

    # Bind submit button and textbox to the respond function
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Run the app
if __name__ == "__main__":
    demo.launch()