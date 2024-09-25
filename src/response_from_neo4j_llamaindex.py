import os
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jGraphStore, Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
load_dotenv()

def response_from_neo4j(query):
    # Neo4j credentials SLT
    # NEO4J_URI = os.getenv("NEO4J_URI")
    # DATABASE = os.getenv("NEO4J_DATABASE")
    # PASSWORD = os.getenv("NEO4J_PASSWORD")
    # USERNAME = os.getenv("NEO4J_USERNAME")

    # Neo4j credentials Local
    NEO4J_URI = "neo4j+s://64fd7f08.databases.neo4j.io"
    USERNAME = "neo4j"
    PASSWORD = "VC35fxfaLpAMI47lRaxVipVE-ZJMehJ2tmp06tgQLG4"
    DATABASE = "neo4j"

    Settings.llm= OpenAI(model= "gpt-4o", temperature=0.1)
    Settings.embed_model= OpenAIEmbedding()
    graph_store = Neo4jPropertyGraphStore(
        username=USERNAME,
        password=PASSWORD,
        url=NEO4J_URI,
    )

    index= PropertyGraphIndex.from_existing(
                                    graph_store, 
                                    embed_model=Settings.embed_model,  
                                    kg_extractors=[SchemaLLMPathExtractor(llm=Settings.llm)], 
                                    show_progress=True
                                )



    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        index=index,
        storage_context=storage_context,
        verbose=True,
    )

    query_engine = RetrieverQueryEngine.from_args(
        graph_rag_retriever,
    )

    #add a prompt
    response = query_engine.query(query)
    return response

query = "Who directed most movies?"
response = response_from_neo4j(query)
print(response)