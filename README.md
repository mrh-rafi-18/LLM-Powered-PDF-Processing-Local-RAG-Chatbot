# 📄 LLM-Powered PDF RAG Chatbot  

An intelligent **PDF Question-Answering Project** built using **LangChain**, **Streamlit**, and **local/remote LLMs**.  
This project allows you to upload a PDF and interact with it in natural language.  

---

## ✨ Features  

- Upload any PDF and ask questions interactively  
- Extracts **text, tables, and embedded images** using **Qwen3-VL-235B-Instruct** (via Hugging Face API)  
- Splits and processes content into chunks for efficient retrieval  
- Generates embeddings with **intfloat/multilingual-e5-small** stored in **FAISS**  
- Retrieves and answers queries using **Llama 3.2 (via Ollama)** with Retrieval-Augmented Generation (RAG)  
- Simple and interactive **Streamlit UI** for user-friendly interaction  

---

## ⚡ Setup Instructions  

⚠️ **Caution:** Since the RAG system is executing locally, it may take a long time to execute.  

### 1. Download and Install Ollama  
- Download the `OllamaSetUp.exe`   
- Install Ollama on your system  

### 2. Pull the Model  
```bash
ollama pull llama3.2
```  
*Note: This may take some time depending on your internet speed*  

### 3. Unzip Project Files  
- Install the zip file and unzip it

### 4. Open Project in VS Code  
- Launch VS Code  
- Open the folder named **"LLM Powered PDF RAG Chatbot"**  

### 5. Create a Virtual Environment  
```bash
python -m venv venv
```

### 6. Activate the Virtual Environment  
```bash
venv\Scripts\activate
```

### 7. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 8. Set Hugging Face API Token  
- Open the `.env` file  
- Provide your **Hugging Face API token** inside it  

### 9. Run the Application  
```bash
streamlit run app.py
```

---

## 📝 PDF Processing  

- To observe how PDFs are processed, open the provided **notebook file** in Google Colab  
- Run the notebook **cell by cell** to see the step-by-step PDF processing workflow  

---

## 🚀 Tech Stack  

- **LangChain** – Orchestration framework  
- **Streamlit** – Simple UI  
- **Qwen3-VL-235B-Instruct (via Hugging Face API)** – Multimodal PDF text + image extraction  
- **intfloat/multilingual-e5-small** – Embeddings  
- **FAISS** – Vector storage & retrieval  
- **Llama 3.2 (via Ollama)** – Local inference with RAG  

---

## 📂 Repository  

🔗 GitHub Repository : https://github.com/mrh-rafi-18/LLM-Powered-PDF-Processing-Local-RAG-Chatbot.git 

---

## 📸 Demo / Screenshots

Here’s how the project looks:

**1. Simple UI of the PDF RAG Chatbot**  
![UI Screenshot](Demo%20Screenshots/ui.png)

**2. PDF Upload and Processing**  
![PDF Processing Screenshot](Demo%20Screenshots/pdf%20processing.png)

**3. Inference / Chat with PDF**  
![Inference Screenshot](Demo%20Screenshots/inference.png)



