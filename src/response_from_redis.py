import os
import re
from dotenv import load_dotenv

# LlamaIndex modules
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline, IngestionCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from redisvl.schema import IndexSchema

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Redis connection details
redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')
redis_password = os.getenv('REDIS_PASSWORD')

Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)

# Function to clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text)

# Function to load documents
def load_documents(data_folder):
    reader = SimpleDirectoryReader(
        input_dir=data_folder,  
        recursive=True
    )
    documents = []
    for docs in reader.iter_data():
        for doc in docs:
            documents.append(doc)
    cleaned_documents = [clean_text(doc.text) for doc in documents]
    return Document(text="\n\n".join([doc for doc in cleaned_documents]))

# Function to create Redis index (Vector and Document Store)
def create_redis_store():
    # Create embedding model
    embed_model = OpenAIEmbedding()

    custom_schema = IndexSchema.from_dict(
        {
        # customize basic index specs
        "index": {
            "name": "tmfp2_index",
            "prefix": "tmforum",
        },
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom metadata fields
            {"type": "numeric", "name": "updated_at"},
            {"type": "tag", "name": "file_name"},
            # custom vector field definition for cohere embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 1536,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
        })
    
    # # Redis vector store
    vector_store = RedisVectorStore(
    schema=custom_schema,  # provide customized schema
    redis_url=f"redis://{redis_host}:{redis_port}",
    overwrite=False
    )
    # Redis document store
    doc_store = RedisDocumentStore.from_host_and_port(
        redis_host, redis_port, namespace="tmfp2_index",
    )
    
    # Redis cache
    cache = IngestionCache(
        cache=RedisCache.from_host_and_port(redis_host, redis_port),
        collection="tmfp2_redis_cache"
    )
    
    return vector_store, doc_store, cache, embed_model

# Function to store embeddings in Redis (Run once during setup)
def store_embeddings(document, vector_store, doc_store, cache, embed_model):
    # Ingestion pipeline setup
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), embed_model],
        docstore=doc_store,
        vector_store=vector_store,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS
    )

    # Index the document
    index = VectorStoreIndex.from_vector_store(pipeline.vector_store, embed_model=embed_model)
    nodes = pipeline.run(documents=[document])
    print(f"Ingested {len(nodes)} Nodes")
    return index

# Function to create query engine (For frequent querying)
def create_query_engine(vector_store, embed_model):
    # Initialize storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index from existing vector store
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, storage_context=storage_context)

    # Create and return query engine
    query_engine = index.as_query_engine()
    return query_engine
