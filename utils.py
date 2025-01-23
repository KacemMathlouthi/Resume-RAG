from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

def setup_embedding_model(model_name: str):
    """
    Setup embedding model for RAG chatbot.
    Returns:
        Embedding_model: HuggingFaceEmbeddings object.
    """
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model

def load_text_data(file_path: str):
    """
    Load text data from a file.
    Args:
        file_path (str): Path to the file.
    Returns:
        data (list): List of strings representing the text data.
    """
    with open(file_path, "r") as file:
        data = file.read()
    return data

def semantic_search(prompt, retriever):
    """
    Perform semantic search using the retriever.

    :param prompt: str, The user query or prompt.
    :param retriever: retriever object, Initialized retriever for the database.
    :return: List of relevant documents.
    """
    # Query the retriever with the user prompt
    results = retriever.get_relevant_documents(prompt)
    final = ""
    # Display the results
    for idx, doc in enumerate(results, 1):
        final += doc.page_content + "\n\n"
    
    return final


def resume_chat_completion(client, model, user_question, relevant_excerpts):
    # Define the system prompt
    system_prompt = '''
    You are an assistant in Kacem Mathlouthi's Portfolio, trying to answer their queries about him. Given the user's question and relevant excerpts from 
    Kacem's resume, answer the question factually. If the question is not available to the exerpt or the question is not relevant to Kacem Mathlouthi, you don't answer it. 
    '''

    # Generate a response to the user's question using the pre-trained model
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content":  system_prompt
            },
            {
                "role": "user",
                "content": "User Question: " + user_question + "\n\Relevant Kacem's Resume/CV Exerpt(s):\n\n" + relevant_excerpts,
            }
        ],
        model = model
    )

    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content

    return response

