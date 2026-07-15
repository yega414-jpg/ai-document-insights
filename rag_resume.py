from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import gradio as gr

load_dotenv()

# STEP 1 - LOAD PDF
def load_pdf(file_path):
    print(f"Loading PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(f"Total characters loaded: {len(text)}")
    return text

# STEP 2 - CHUNK THE TEXT
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # each chunk = 500 characters
        chunk_overlap=50,      # 50 chars overlap between chunks
        length_function=len
    )
    chunks = splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nSample chunk:\n{chunks[0]}")
    return chunks

# STEP 3 - CREATE VECTOR STORE
# STEP 3 - CREATE VECTOR STORE
def create_vector_store(chunks):
    print("\nCreating embeddings and storing in ChromaDB...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Use in-memory ChromaDB instead of disk - avoids Windows lock issue
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
        # No persist_directory = stays in memory, no file lock!
    )
    
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")
    return vectorstore  

# STEP 4 - SEARCH THE RESUME
def search_resume(vectorstore, query):
    print(f"\n🔍 Searching for: {query}")
    
    results = vectorstore.similarity_search(query, k=2)  # top 2 matches
    
    print(f"\nTop {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Match {i} ---")
        print(doc.page_content)
        print()
    
    return results

# STEP 5 - ASK A QUESTION and LLM translates to Natural language getting the answer from the retrieved context  
def ask_question(vectorstore, question):
    print(f"\n❓ Question: {question}")
    
    # STEP 1 - RETRIEVAL: find relevant chunks
    results = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    # STEP 2 - GENERATION: send to LLM with context
    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions about the provided document. Only use information from the document context. If the answer is not in the context, say so."
        },
        {
            "role": "user",
            "content": f"""Document context:
{context}

Question: {question}

Answer based only on the document context above."""
        }
    ]
    
    response = client.chat_completion(
        # model="Qwen/Qwen2.5-72B-Instruct",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        max_tokens=300

    )
    
    answer = response.choices[0].message.content.strip()
    print(f"\n💡 Answer: {answer}")
    return answer

# TEST IT
if __name__ == "__main__":
    import gradio as gr

    vectorstore_holder = {"vs": None}

    def upload_document(pdf_file):
        # Bug 1 fix - handle when file is None
        if pdf_file is None:
            return "⚠️ Please select a PDF file."

        text = load_pdf(pdf_file)  # Gradio now passes path directly, not object
        chunks = chunk_text(text)
        vectorstore_holder["vs"] = create_vector_store(chunks)
        return "✅ Document uploaded and ready! Go to the Get Insights tab."

    def chat(question, history):
        if vectorstore_holder["vs"] is None:
            return "⚠️ Please upload a document first in the Upload tab."
        return ask_question(vectorstore_holder["vs"], question)

    with gr.Blocks(title="Document Insights AI") as demo:
        gr.Markdown("# 📚 Document Insights AI")
        gr.Markdown("Upload any PDF document and ask questions to get insights from it.")

        with gr.Tab("📄 Upload Document"):
            upload = gr.File(label="Upload any PDF", file_types=[".pdf"])
            status = gr.Textbox(label="Status")
            upload.change(fn=upload_document, inputs=upload, outputs=status)

        with gr.Tab("💬 Get Insights"):
            gr.ChatInterface(fn=chat)

    demo.launch()