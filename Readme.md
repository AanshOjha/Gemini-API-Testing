## Basics
### 1. Vector
* a vector is a list of numbers that represents data.
* All complex data, image, video, docs are represented by a numerical value.
* Vectors can have hundreds or thousands of dimensions to store the data.

### 2. Embedding
* Way of converting complex data (like words, images, or other forms of content) into vectors. This process is called "embedding" because it embeds the information into a high-dimensional space.
* A vector database stores these vector embeddings.

> Using Qdrant
> 1. Run this in terminal
> ```
> docker pull qdrant/qdrant
> docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant
> ```

### 3. Collections
* It's a container where you store your vectors in qdrant (data represented as lists of numbers) and associated metadata (payload).
* collection is created on the Qdrant server.

### 4. Dimensions
* The dimensions are usually defined by the embedding algorithm or model. 
### 5. Distance Metric: 
* This is a way to measure how similar or different two vectors are. Common distance metrics include:
#### Euclidean Distance: 
Measures the straight-line distance between two points in Euclidean space.
#### Dot Product: 
* Measures the cosine similarity (angle) between two vectors, often used for text embeddings.
* Eg. A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4], the dot product is calculated as: 
* Dot Product = 𝑎1×𝑏1 + 𝑎2×𝑏2 + 𝑎3×𝑏3 + 𝑎4×𝑏4

### 6. Payload
```py
point = PointStruct(id=1, vector=vector.tolist(), payload={"document": "Python Programming"})
```
The payload is a way to store extra information about the data point that can be useful for filtering or understanding the context of the vector.

### 7. Vectorstore
A vector store is used for storing and retrieving high-dimensional vectors.

## For PDF Query Using Gemini API
### Why FastAPI? 
* Asynchronous Capabilities: FastAPI supports asynchronous programming, which allows to execute other endpoints while one is being executed. 
If the application were to be expanded with additional endpoints or background tasks that could be handled concurrently, the use of asynchronous capabilities would become more beneficial.
* Performance: FastAPI is built on ASGI (Asynchronous Server Gateway Interface), making it faster than Flask, which relies on WSGI (Web Server Gateway Interface). 

### class PDFVectorSearchService
* Created to encapsulate all the functionalities related to processing PDFs, creating vector stores, and querying documents. This makes the code modular, organized, and easier to maintain.
### Temperature
```py
llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
)
```
#### Low Temperature (e.g., 0.1):

* It tends to choose the most probable next token, leading to more predictable and repetitive responses.
* Useful for tasks requiring precision and reliability.

#### High Temperature (e.g., 0.9):

* It samples from a broader range of possible next tokens, leading to more varied and imaginative responses.
* Useful for creative writing or generating varied content.

> Default is 0.7



> __No of dimensions in vector, dot or euclidean product to find similarity, converting a data to vector, all these are decided by `Embedding Model`__

### Process
1. Identify the Type of Data
2. Select an Appropriate Embedding Model
3. Check the number of dimensions it produces, Determine the recommended distance metric.
4. convert your data into vectors
5. Store and Query in Vector Database:
* Create a collection in Qdrant.
* Insert the vectors into this collection.
* Use the recommended distance metric for querying.

# Querying PDF with Qdrant and Gemini APIs

### PDF Querying
1. Make sure `Qdrant` is up and running.
```
docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant
```
2. Get GOOGLE_API_KEY, load in .env
3. `pip install -r .\requirements.txt`
4. `python pdf_query.py`
5. `curl -X POST -F "file=@path/to/pdf" http://localhost:8000/upload-pdf`
6. `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "what is python known for?"}'`

### Testing with 800+ pages PDF
![alt text](img/image.png)


> For my app, 
 ```bash
 curl -X POST -F "file=@C:\Users\itsaa\OneDrive\Desktop\PDF_Vector\data\python.pdf" http://localhost:8000/upload-pdf -w "\nTime taken: %{time_total} seconds\n"
 ```

 ```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "Tell about python?"}' -w "\nTime taken: %{time_total} seconds\n"
 ```