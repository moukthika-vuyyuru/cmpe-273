# -*- coding: utf-8 -*-

# You might not need to install these via script, as they are usually installed once manually.
# pip install redis pandas openai
import openai
import numpy as np
import redis
import os
import wget
import zipfile
import pandas as pd
from ast import literal_eval
import ssl
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import time


# Constants
INDEX_NAME = "embeddings-index"  # Name of the search index
PREFIX = "doc"                   # Prefix for the document keys in Redis
VECTOR_DIM = 300   

#OPENAI_API_KEY = 
openai.api_key = "sk-xxx"
# Function to download Wikipedia data
def download_wikipedia_data(data_path: str = './data/', download_path: str = "./",
                            file_name: str = "vector_database_wikipedia_articles_embedded") -> pd.DataFrame:
    data_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
    csv_file_path = os.path.join(data_path, file_name + ".csv")
    zip_file_path = os.path.join(download_path, file_name + ".zip")

    ssl._create_default_https_context = ssl._create_unverified_context
    wget.download(data_url, out=download_path)
    if os.path.isfile(csv_file_path):
        print("File already downloaded.")
    else:
        if not os.path.isfile(zip_file_path):
            print("Downloading zip file...")
            wget.download(data_url, out=download_path)
        
        print("Unzipping data...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)

        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        print(f"Data downloaded and extracted to {data_path}")

# Function to read Wikipedia data
"""def read_wikipedia_data(data_path: str = './data/',
                        file_name: str = "vector_database_wikipedia_articles_embedded",
                        nrows: int = None) -> pd.DataFrame:
    csv_file_path = os.path.join(data_path, file_name + ".csv")
    data = pd.read_csv(csv_file_path, nrows=nrows)
    data['title_vector'] = data['title_vector'].apply(literal_eval)
    data['content_vector'] = data['content_vector'].apply(literal_eval)
    data['vector_id'] = data['vector_id'].apply(str)
    return data"""

def read_wikipedia_data(data_path: str = './data/',
                        file_name: str = "vector_database_wikipedia_articles_embedded") -> pd.DataFrame:
    csv_file_path = os.path.join(data_path, file_name + ".csv")
    data = pd.read_csv(csv_file_path)
    data['title_vector'] = data['title_vector'].apply(literal_eval)
    data['content_vector'] = data['content_vector'].apply(literal_eval)
    data['vector_id'] = data['vector_id'].apply(str)
    return data
# Function to create a hybrid search field
def create_hybrid_field(field_name: str, value: str) -> str:
    return f'@{field_name}:"{value}"'

# Function to run hybrid queries with Redis
def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "embeddings-index",
    vector_field: str = "title_vector",
    return_fields: list = ["title", "url", "text", "vector_score"],
    hybrid_fields="*",
    k: int = 20,
    print_results: bool = True,
) -> list[dict]:

    # Creates embedding vector from user query using OpenAI
    embedded_query = openai.Embedding.create(input=user_query,
                                            model="text-embedding-ada-002",
                                            )["data"][0]['embedding']

    # Prepare the Query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    query = (
        Query(base_query)
         .return_fields(*return_fields)
         .sort_by("vector_score")
         .paging(0, k)
         .dialect(2)
    )
    params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}

    # Perform vector search
    results = redis_client.ft(index_name).search(query, params_dict)
    if print_results:
        for i, article in enumerate(results.docs):
            score = 1 - float(article.vector_score)
            print(f"{i}. {article.title} (Score: {round(score, 3)})")
    return results.docs

def create_search_index(redis_client, data):
    print("Creating search index...")
    VECTOR_DIM = len(data['title_vector'][0])  # Length of the vectors
    VECTOR_NUMBER = len(data)                  # Initial number of vectors
    INDEX_NAME = "embeddings-index"            # Name of the search index
    PREFIX = "doc"                             # Prefix for the document keys
    DISTANCE_METRIC = "COSINE"                 # Distance metric for the vectors

    # Define RediSearch fields
    title = TextField(name="title")
    url = TextField(name="url")
    text = TextField(name="text")
    title_embedding = VectorField("title_vector", "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    })
    text_embedding = VectorField("content_vector", "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    })
    fields = [title, url, text, title_embedding, text_embedding]

    # Create RediSearch Index
    try:
        redis_client.ft(INDEX_NAME).info()
        print("Index already exists")
    except:
        redis_client.ft(INDEX_NAME).create_index(
            fields=fields,
            definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
        )
        print("Search index created")
def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    records = documents.to_dict("records")
    indexed_count = 0  # Counter for successfully indexed documents

    for doc in records:
        key = f"{prefix}:{str(doc['vector_id'])}"
        
        # Convert embeddings to byte arrays
        title_embedding = np.array(doc["title_vector"], dtype=np.float32).tobytes()
        content_embedding = np.array(doc["content_vector"], dtype=np.float32).tobytes()

        # Create a new dictionary for Redis, ensuring all data is in a compatible format
        redis_doc = {
            'url': str(doc['url']),
            'title': str(doc['title']),
            'text': str(doc['text']),
            'title_vector': title_embedding,
            'content_vector': content_embedding
        }

        try:
            client.hset(key, mapping=redis_doc)
            indexed_count += 1
        except Exception as e:
            print(f"Failed to index document {key}: {e}")

    print(f"Indexed {indexed_count} documents out of {len(documents)}")

def time_queries(redis_client, iterations: int = 10):
    print(" ----- Flat Index ----- ")
    t0 = time.time()
    for i in range(iterations):
        results_flat = search_redis(redis_client, 'modern art in Europe', k=10, print_results=False)
    t0 = (time.time() - t0) / iterations
    results_flat = search_redis(redis_client, 'modern art in Europe', k=10, print_results=True)
    print(f"Flat index query time: {round(t0, 3)} seconds\n")
    time.sleep(1)
    print(" ----- HNSW Index ------ ")
    t1 = time.time()
    for i in range(iterations):
        results_hnsw = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10, print_results=False)
    t1 = (time.time() - t1) / iterations
    results_hnsw = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10, print_results=True)
    print(f"HNSW index query time: {round(t1, 3)} seconds")
    print(" ------------------------ ")



def main():
    # Local Redis Connection (change these settings as per your local Redis setup)
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_PASSWORD = ""  # Add password here if needed

    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
        print("Connection successful:", redis_client.ping())
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    # Check if data file exists
    data_file_path = './data/vector_database_wikipedia_articles_embedded.csv'
    if not os.path.isfile(data_file_path):
        # Download and read Wikipedia data if not already present
        download_wikipedia_data()
        data = read_wikipedia_data()
    else:
        # Load data from the existing file
        data = read_wikipedia_data()

    print(data.head())

    # Create search index in Redis
    create_search_index(redis_client, data)

    index_documents(redis_client, PREFIX, data)
    # Print the number of documents loaded
    db_info = redis_client.info('db0')
    print(f"Loaded {db_info.get('keys', 0)} documents in Redis search index with name: {INDEX_NAME}")

    # Run a hybrid query combining vector search with text search
    text_query = "Famous battles in Scottish history"
    hybrid_query_results = search_redis(
        redis_client,
        text_query,
        vector_field="title_vector",
        k=5,
        hybrid_fields=create_hybrid_field("title", "Scottish")
    )

    # Print the results of the hybrid query
    print("Hybrid Search Results 1:")
    for i, article in enumerate(hybrid_query_results):
        score = 1 - float(article.vector_score)
        print(f"{i}. {article.title} (Score: {round(score, 3)})")

    # Run another hybrid query
    text_query = "Art"
    hybrid_query_results = search_redis(
        redis_client,
        text_query,
        vector_field="title_vector",
        k=5,
        hybrid_fields=create_hybrid_field("text", "Leonardo da Vinci")
    )

    # Print the results of the second hybrid query
    print("Hybrid Search Results 2:")
    for i, article in enumerate(hybrid_query_results):
        score = 1 - float(article.vector_score)
        print(f"{i}. {article.title} (Score: {round(score, 3)})")

    # Find a specific mention of "Leonardo da Vinci" in the text returned by the second query
    mention = [sentence for sentence in hybrid_query_results[0].text.split("\n") if "Leonardo da Vinci" in sentence][0]
    print(f"Mention of 'Leonardo da Vinci': {mention}")

    # Compare query performance between HNSW and FLAT indices
    time_queries(redis_client)

if __name__ == "__main__":
    main()
