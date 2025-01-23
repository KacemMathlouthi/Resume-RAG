# **RAG Resume Chatbot with Gradio**
A chatbot application that leverages **retrieval-augmented generation (RAG)** to answer questions about an individual's professional experience. The chatbot is designed to retrieve relevant information from a resume and generate intelligent responses using large language models (LLMs).

---

## **Features**
- **Resume-based Question Answering:** Ask questions about Kacem Mathlouthi's professional background.
- **Semantic Search:** Uses embeddings to retrieve the most relevant sections of the resume.
- **Gradio Interface:** A user-friendly interface for interacting with the chatbot.

---

## **Technologies Used**
- **Python**: Core programming language for the project.
- **Gradio**: For building the chatbot interface.
- **LangChain**: To handle embeddings and retrieval-augmented search.
- **Chroma**: For vector-based semantic search.
- **Groq**: For generating responses using pre-trained language models.
- **sentence-transformers**: For embedding generation.

---

## **Project Structure**

```
.
├── LICENSE              # License for the project
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── app.py               # Main application script
├── utils.py             # Utility functions for embeddings, search, etc.
├── data
│   └── resume.txt           # Resume data for answering questions
└── .env                 # Environment variables (You should create this file)
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/KacemMathlouthi/Resume-RAG.git
cd Resume-RAG
```

### **2. Create a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies.
```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows
```

### **3. Install Dependencies**
Install all required libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### **4. Add the Environment Variables**
Create a `.env` file in the root directory and add the following:
```
GROQ_API_KEY=your_groq_api_key_here
```

### **5. Run the Application**
Launch the Gradio interface:
```bash
python app.py
```
The application will run locally, and you can access it in your browser at [http://127.0.0.1:7860](http://127.0.0.1:7860).

---

## **Usage**
**Ask Questions:** Use the chatbot interface to ask specific questions about the resume. For example:
   - *"What are Kacem Mathlouthi's main skills?"*
   - *"Tell me about his previous work experience."*

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
