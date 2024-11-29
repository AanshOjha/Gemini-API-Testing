import os
import csv
from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from qdrant_client import QdrantClient
import dotenv
from langchain.schema import Document

# Load environment variables
dotenv.load_dotenv()

# Class to make code modular, organized
class CSVVectorSearchService:
    def __init__(self, GOOGLE_API_KEY):
        # Configure Gemini API
        genai.configure(api_key=GOOGLE_API_KEY)

        # Create embeddings from Google Generative AI
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Initialize Qdrant Vector Store
        self.qdrant_url = "http://localhost:6333"  # Local Qdrant instance

# Create vector store from CSV
    def process_csv(self, file_path):
    # Read CSV file with full context preservation
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # Create documents that preserve column names and row data
            docs = []
            for row in reader:
                # Create a comprehensive text representation of the row
                # Include column names and values to maintain context
                row_text = " ".join([
                    f"{key}: {value}" for key, value in row.items()
                ])

                # Create a Document with the full row context
                doc = Document(
                    page_content=str(row),
                    metadata=row  # Store the entire row as metadata
                )
                docs.append(doc)

        # # Split documents more carefully
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500,  # Smaller chunk size to preserve row context
        #     chunk_overlap=100
        # )
        # split_docs = text_splitter.split_documents(docs)

        # Create a unique collection name
        global unique_collection_name
        unique_collection_name = f"csv_documents_{os.path.basename(file_path)}"

        # Store document embeddings in Qdrant
        vectorstore = Qdrant.from_documents(
            docs, 
            self.embeddings,
            url=self.qdrant_url,
            collection_name=unique_collection_name
        )

        print(docs)

        return vectorstore


    def query_documents(self, query, vectorstore):
    # Initialize Gemini Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        # Custom Prompt Template with more context
        prompt_template = """
        Use the following context to precisely answer the question. 
        Each context entry contains full row information from the CSV.
        If the exact answer is not found, explain why.

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
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,  # Increase to get more context
                    "filter": None  # Remove any default filtering
                }
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Execute Query
        result = retrieval_qa.invoke(query)
        return result['result']

# FastAPI Application
app = FastAPI()

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    filename = str(file.filename)
    with open(filename, "wb") as buffer:
        buffer.write(await file.read())

    # Initialize service
    service = CSVVectorSearchService(
        GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    )

    # Process CSV and create vector store
    vectorstore = service.process_csv(file.filename)

    # Clean up temporary file
    os.remove(filename)

    return {"status": "CSV processed and vectorized"}

# Update the query endpoint in main.py
@app.post("/query")
async def query_documents(query: dict):
    try:
        # Extract the query text from the JSON
        query_text = query.get('query')
        
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")

        # Initialize service
        service = CSVVectorSearchService(
            GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
        )

        # Load existing Qdrant vector store
        vectorstore = Qdrant(
            client=QdrantClient(url="http://localhost:6333"), 
            embeddings=service.embeddings,
            collection_name=unique_collection_name
        )

        # Query documents 
        result = service.query_documents(query_text, vectorstore)

        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uvicorn for FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)