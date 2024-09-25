from response_from_neo4j import handle_user_query
from response_from_redis import create_query_engine,create_redis_store

user_query = input("Ask a query: ")
response_neo4j = handle_user_query(user_query)

vector_store, doc_store, cache, embed_model = create_redis_store()
query_engine = create_query_engine(vector_store,embed_model)
response_redis  = query_engine.query(user_query)
print(response_redis)