import os
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jGraphStore, Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext

load_dotenv()

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

llm= OpenAI(model= "gpt-4o", temperature=0.1)
embed_model= OpenAIEmbedding()
graph_store = Neo4jPropertyGraphStore(
    username=USERNAME,
    password=PASSWORD,
    url=NEO4J_URI,
)

index= PropertyGraphIndex.from_existing(
                                   graph_store, 
                                   embed_model=embed_model,  
                                   kg_extractors=[SchemaLLMPathExtractor(llm=llm)], 
                                   show_progress=True
                              )


from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
storage_context = StorageContext.from_defaults(graph_store=graph_store)

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
)

response = query_engine.query("Who directed most movies?")
print(response)