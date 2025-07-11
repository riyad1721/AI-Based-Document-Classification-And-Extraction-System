import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from text_extraction import extract_text_from_image
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN not found in environment variables.")

VECTOR_DB_DIR = "faiss_index"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 50

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def clean_text(text):
    lines = text.splitlines()
    filtered = [line for line in lines if len(line.strip()) > 5 and not line.strip().startswith(("---", "Page"))]
    return "\n".join(filtered)

def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def create_vectorstore(docs):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    db = FAISS.from_documents(docs, embedding)
    db.save_local(VECTOR_DB_DIR)
    return db

def setup_qa_chain(vectorstore):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small", use_auth_token=hf_token)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", use_auth_token=hf_token)
    
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant for financial documents.\n"
            "Based only on the following context, answer the question briefly.\n"
            "If the answer cannot be found, just respond with 'Not found'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
