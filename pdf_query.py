import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai

import dotenv
from qdrant_client import QdrantClient

dotenv.load_dotenv()
# Requirements
"""
fastapi
uvicorn
langchain
langchain-community
langchain-google-genai
qdrant-client
google-generativeai
python-multipart
PyPDF2
"""

class PDFVectorSearchService:
    def __init__(self, gemini_api_key):
        # Configure Gemini API
        os.environ['GOOGLE_API_KEY'] = gemini_api_key
        genai.configure(api_key=gemini_api_key)

        # Initialize Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        # Initialize Qdrant Vector Store
        self.qdrant_url = "http://localhost:6333"  # Local Qdrant instance
        self.collection_name = "pdf_documents"

    def process_pdf(self, file_path):
        # Use Langchain PDF Loader
        loader = PyPDFLoader(file_path)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = loader.load_and_split(text_splitter)

        # Create Vector Store
        vectorstore = Qdrant.from_documents(
            docs, 
            self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name
        )

        return vectorstore

    def query_documents(self, query, vectorstore):
        # Initialize Gemini Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        # Custom Prompt Template
        prompt_template = """
        Use the following context to answer the question. 
        If the answer is not in the context, say "I cannot find the answer in the provided documents."

        Context: {context}
        Question: {question}
        
        Helpful Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        # Create Retrieval QA Chain
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Execute Query
        result = retrieval_qa.invoke(query)
        return result['result']

# FastAPI Application
app = FastAPI()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # Initialize service
    service = PDFVectorSearchService(
        gemini_api_key=os.getenv('GOOGLE_API_KEY')
    )

    # Process PDF and create vector store
    vectorstore = service.process_pdf(file.filename)

    # Clean up temporary file
    os.remove(file.filename)

    return {"status": "PDF processed and vectorized"}

# Update the query endpoint in main.py
@app.post("/query")
async def query_documents(query: dict):
    try:
        # Extract the query text from the JSON
        query_text = query.get('query')
        
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")

        # Initialize service
        service = PDFVectorSearchService(
            gemini_api_key=os.getenv('GOOGLE_API_KEY')
        )

        # Load existing Qdrant vector store
        vectorstore = Qdrant(
            client=QdrantClient(url="http://localhost:6333"), 
            embeddings=service.embeddings,
            collection_name=service.collection_name
        )

        # Query documents 
        result = service.query_documents(query_text, vectorstore)

        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional configurations can be added here
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)