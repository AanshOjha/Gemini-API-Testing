# Basics
### 1. Vector
* a vector is a list of numbers that represents data.
* All complex data, image, video, docs are represented by a numerical value.
* Vectors can have hundreds or thousands of dimensions to store the data.

### 2. Embedding
* Way of converting complex data (like words, images, or other forms of content) into vectors. This process is called "embedding" because it embeds the information into a high-dimensional space.
* A vector database stores these vector embeddings.

## Using Qdrant
1. Run this in terminal
```
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

### Collections
* It's a container where you store your vectors in qdrant (data represented as lists of numbers) and associated metadata (payload).
* collection is created on the Qdrant server.

### Dimensions
* The dimensions are usually defined by the embedding algorithm or model. 
### Distance Metric: 
* This is a way to measure how similar or different two vectors are. Common distance metrics include:
#### Euclidean Distance: 
Measures the straight-line distance between two points in Euclidean space.
#### Dot Product: 
* Measures the cosine similarity (angle) between two vectors, often used for text embeddings.
* Eg. A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4], the dot product is calculated as: 
* Dot Product = 𝑎1×𝑏1 + 𝑎2×𝑏2 + 𝑎3×𝑏3 + 𝑎4×𝑏4

### Payload
```py
point = PointStruct(id=1, vector=vector.tolist(), payload={"document": "Python Programming"})
```
The payload is a way to store extra information about the data point that can be useful for filtering or understanding the context of the vector.

### Vector store
A vector store is used for storing and retrieving high-dimensional vectors.

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

# PDF Querying
1. Get GOOGLE_API_KEY, load in .env
2. `pip install -r .\requirements.txt`
3. 
4. `curl -X POST -F "file=@path/to/pdf" http://localhost:8000/upload-pdf`
5. `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "what is python known for?"}'`