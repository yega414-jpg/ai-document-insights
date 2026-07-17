from dotenv import load_dotenv
import os
import io
import boto3
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import gradio as gr

load_dotenv()

# S3 CLIENT
def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )

# STEP 1 - LOAD PDF FROM LOCAL PATH
def load_pdf(file_path):
    print(f"Loading PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(f"Total characters loaded: {len(text)}")
    return text

# STEP 1B - LOAD PDF FROM S3
def load_pdf_from_s3(s3_key):
    print(f"Loading PDF from S3: {s3_key}")
    s3 = get_s3_client()
    response = s3.get_object(
        Bucket=os.getenv('AWS_BUCKET_NAME'),
        Key=s3_key
    )
    pdf_bytes = response['Body'].read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(f"Total characters loaded: {len(text)}")
    return text

# STEP 2 - CHUNK THE TEXT
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nSample chunk:\n{chunks[0]}")
    return chunks

# STEP 3 - CREATE VECTOR STORE
def create_vector_store(chunks):
    print("\nCreating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")
    return vectorstore

# STEP 4 - ASK A QUESTION
# STEP 4 - ASK A QUESTION
def ask_question(vectorstore, question):
    print(f"\n❓ Question: {question}")
    results = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant.
Answer using only the document context below.
If answer not in context, say so.

Context: {context}

Question: {question}

Answer:"""
    )

    chain = prompt | llm
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"\n💡 Answer: {answer.content}")
    return answer.content

# LIST FILES IN S3
def list_s3_files():
    try:
        s3 = get_s3_client()
        response = s3.list_objects_v2(
            Bucket=os.getenv('AWS_BUCKET_NAME')
        )
        if 'Contents' not in response:
            return "No files in S3 bucket yet"
        files = [obj['Key'] for obj in response['Contents']
                 if obj['Key'].endswith('.pdf')]
        return "\n".join(files) if files else "No PDFs found"
    except Exception as e:
        return f"Error: {str(e)}"

# GRADIO UI
if __name__ == "__main__":

    vectorstore_holder = {"vs": None}

    # Upload local PDF → save to S3 → load from S3
    def upload_document(pdf_file):
        if pdf_file is None:
            return "⚠️ Please select a PDF file."
        try:
            # Get filename
            file_name = os.path.basename(pdf_file)
            s3_key = f"documents/{file_name}"

            # STEP 1 - Upload to S3
            print(f"Uploading {file_name} to S3...")
            s3 = get_s3_client()
            s3.upload_file(
                pdf_file,
                os.getenv('AWS_BUCKET_NAME'),
                s3_key
            )
            print(f"✅ Uploaded to S3: {s3_key}")

            # STEP 2 - Read from S3
            text = load_pdf_from_s3(s3_key)

            # STEP 3 - Create vector store
            chunks = chunk_text(text)
            vectorstore_holder["vs"] = create_vector_store(chunks)

            return f"✅ Uploaded to S3 and ready!\nSaved as: {s3_key}"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    # Load existing file from S3
    def load_from_s3(s3_key):
        if not s3_key:
            return "⚠️ Please enter an S3 file key"
        try:
            text = load_pdf_from_s3(s3_key.strip())
            chunks = chunk_text(text)
            vectorstore_holder["vs"] = create_vector_store(chunks)
            return f"✅ Loaded from S3: {s3_key}"
        except Exception as e:
            return f"❌ Error loading from S3: {str(e)}"

    # Chat
    def chat(question, history):
        if vectorstore_holder["vs"] is None:
            return "⚠️ Please upload a document first."
        return ask_question(vectorstore_holder["vs"], question)

    with gr.Blocks(title="Document Insights AI") as demo:
        gr.Markdown("# 📚 Document Insights AI")
        gr.Markdown("Upload any PDF — automatically saved to AWS S3")

        with gr.Tab("📄 Upload Document"):
            gr.Markdown("Upload a PDF — it will be saved to S3 automatically")
            upload = gr.File(label="Upload any PDF", file_types=[".pdf"])
            status1 = gr.Textbox(label="Status")
            upload.change(
                fn=upload_document,
                inputs=upload,
                outputs=status1
            )

        with gr.Tab("☁️ Load from S3"):
            gr.Markdown("Load a previously uploaded document from S3")
            s3_files = gr.Textbox(
                label="Files in your S3 bucket",
                value=list_s3_files(),
                interactive=False
            )
            refresh_btn = gr.Button("🔄 Refresh file list")
            refresh_btn.click(fn=list_s3_files, outputs=s3_files)
            s3_input = gr.Textbox(
                label="S3 File Key",
                placeholder="e.g. documents/Yegammai_Ramu.pdf"
            )
            load_btn = gr.Button("Load from S3", variant="primary")
            status2 = gr.Textbox(label="Status")
            load_btn.click(
                fn=load_from_s3,
                inputs=s3_input,
                outputs=status2
            )

        with gr.Tab("💬 Get Insights"):
            gr.ChatInterface(fn=chat)

    demo.launch()