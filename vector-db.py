import os
import redis
import openai
import wikipedia
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

def create_search_index(redis_client):
    try:
        index_name = "wiki_index"

        # Check if the index already exists
        try:
            if redis_client.ft(index_name).info():
                print(f"Index '{index_name}' already exists.")
                return index_name
        except redis.exceptions.ResponseError:
            # Index does not exist, proceed to create it
            pass

        vector_field = VectorField(
            "embedding",
            algorithm="FLAT",
            attributes={
                "TYPE": "FLOAT32",
                "DIM": 128,
                "DISTANCE_METRIC": "COSINE"
            }
        )
        redis_client.ft(index_name).create_index(
            fields=[TextField("title"), vector_field],
            definition=IndexDefinition(prefix=['wiki:'], index_type=IndexType.HASH)
        )
        print(f"Search index '{index_name}' created.")
        return index_name
    except Exception as e:
        print(f"Error creating/searching for search index: {e}")
        return None


# Function to download Wikipedia summary
def download_wikipedia_summary(title):
    try:
        summary = wikipedia.summary(title)
        print(f"Downloaded Wikipedia summary for {title}")
        return summary
    except Exception as e:
        print(f"Error downloading Wikipedia summary for {title}: {e}")
        return None

# Function to connect to Redis
def connect_to_redis():
    try:
        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        print("Connected to Redis")
        return client
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return None

# Function to generate embeddings using OpenAI
def generate_embedding(text):
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-similarity-babbage-001"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Modify the store_embedding function to accept the key argument
def store_embedding(redis_client, key, embedding):
    try:
        redis_client.set(key, str(embedding))
        print(f"Stored embedding for key: {key}")
    except Exception as e:
        print(f"Error storing embedding: {e}")

# Function to retrieve embeddings from Redis
def retrieve_embedding(redis_client, key):
    try:
        embedding = eval(redis_client.get(key))
        print(f"Retrieved embedding for key: {key}")
        return embedding
    except Exception as e:
        print(f"Error retrieving embedding: {e}")
        return None
    
# Load document into the index
def load_document(redis_client, index_name, title, embedding):
    try:
        doc_key = f"wiki:{title.replace(' ', '_')}"
        redis_client.hset(doc_key, mapping={"title": title, "embedding": embedding})
        print(f"Document for '{title}' loaded into index '{index_name}'.")
    except Exception as e:
        print(f"Error loading document into index: {e}")
# Function to compute cosine similarity
def cosine_similarity(vec_a, vec_b):
    return 1 - distance.cosine(vec_a, vec_b)

# Function to perform vector search
def vector_search_query(redis_client, index_name, query_text):
    try:
        query_embedding = generate_embedding(query_text)
        if query_embedding is None:
            raise ValueError("Failed to generate embedding for the query.")

        # Fetch all documents from the index (for demonstration; optimize for production use)
        docs = redis_client.hgetall(index_name)
        similar_docs = []
        for key, doc_embedding_str in docs.items():
            doc_embedding = np.array(eval(doc_embedding_str))
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similar_docs.append((key, similarity))

        # Sort documents by similarity
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        return similar_docs
    except Exception as e:
        print(f"Error during vector search query: {e}")
        return []
def vector_search(redis_client, index_name, query):
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        raise ValueError("Failed to generate embedding for the query.")
    return vector_search_query(redis_client, index_name, query)

# Main execution
def main():
    # Set OpenAI API key
    openai.api_key = "sk-"
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

    # Connect to Redis
    redis_client = connect_to_redis()
    if not redis_client:
        return
    # Create a search index and get the index name
    index_name = create_search_index(redis_client)
    if not index_name:
        return

    # List of topics to fetch summaries for
    topics = ["Artificial Intelligence", "Data Science","Deep Learning"]

    for topic in topics:
        summary = download_wikipedia_summary(topic)
        if summary:
            embedding = generate_embedding(summary)
            if embedding:
                store_embedding(redis_client, f"wiki:{topic.replace(' ', '_')}", embedding)



    # Example query
    query_text = "Neural Networks"
    similar_documents = vector_search(redis_client, index_name, query_text)
    print("Similar Documents:", similar_documents)

if __name__ == "__main__":
    main()
