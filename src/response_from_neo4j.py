import os
from openai import OpenAI as opai
from neo4j import GraphDatabase
from dotenv import load_dotenv
import openai
import json

load_dotenv()
# Set up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = opai()

# Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
DATABASE = os.getenv("NEO4J_DATABASE")
PASSWORD = os.getenv("NEO4J_PASSWORD")
USERNAME = os.getenv("NEO4J_USERNAME")


# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(USERNAME, PASSWORD))
     
# Function to execute Cypher query
def execute_cypher_query(cypher_query):
    single_line_query = ' '.join(cypher_query.splitlines())
    single_line_query = single_line_query.replace('cypher', '')
    single_line_query = single_line_query.replace("'''", '').replace('```', '')
    cypher_query = single_line_query.strip()

    # Print the result
    print(cypher_query)
  
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record for record in result]
        

# Function to generate Cypher query using OpenAI
def generate_cypher_query(prompt):
    # Prepare prompt with schema information and user input
    
    response = client.chat.completions.create(
                model="gpt-4o",
                messages= [{"role": "user", "content": prompt}],
                max_tokens=512
            )
            

    cypher_query = response.choices[0].message.content.strip()
    return cypher_query

# Load JSON file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Generate prompt with reference to JSON
def create_prompt(user_input,json_summary):
    prompt = f"""
    You are an expert at converting natural language queries into Cypher queries.
    Convert the following natural language query into a valid Cypher query that can be directly executed with the Neo4j Python driver session.run function :{user_input}
    Ensure the query syntax is correct and follows Neo4j Cypher best practices, using RETURN for the results. 
    To perform the correlation you will traverse the database and retrieve the data required to answer the user questions. For that,
    you will generate readonly neo4j cypher queries with the help of a STRICT ontology which describes the RAN entities and the relations between them. 
    Then you will execute this query with the help of the provided tools and use the retrieved data to formulate your answer. 
    Query the topology only if you have sufficient details. 
    If the question is unclear, ask the user for more details.
    The RAN network ontology is described by the following JSON object:{json_summary}
    Below are several examples: 
    * To check the Events experienced by a BaseStation, use the Event - EXPERIENCED_BY -> BaseStation relations chain.
    * To correlate events at Site level, traverse the Event - EXPERIENCED_BY -> BaseStation - BELONGS_TO -> Site relations chain.
    * To correlate events at Area level, traverse the Event - EXPERIENCED_BY -> BaseStation - BELONGS_TO -> Site -> BELONGS_TO -> Area relatikons chain. 
    *  To correlate events at EMS level, traverse the Event - EXPERIENCED BY → BaseStation - MANAGED BY → EMS relations.
    Below are some Sample queries:
    * Get EMS vise alarm count details:MATCH (ems:EMS) OPTIONAL MATCH (event:Event)-[:EXPERIENCED_BY]->(bs:BaseStation)-[:MANAGED_BY]->(ems) WHERE event.type = 'alarm' RETURN DISTINCT ems.vendor AS EMSName, COUNT(event) AS NumberOfAlarms ORDER BY ems.vendor
    * To get number of alarms experienced by base station PALAYAN-L4016 :MATCH (event:Event type: 'alarm')-[:EXPERIENCED_BY]->(bs:BaseStation name: 'PALAYAN-L4016')
    RETURN COUNT(event) AS NumberOfAlarms
    Searches inside name and description fields for matching keywords should be case insensitive
    In the case where name field is not present refer to vendor field.
    Always use the full text search index named "eventSearchIndex" present in the neo4j database to search for events that match certain keywords.
    For example, to look for power failure events, use the statement:
    CALL db.index.fulltext.queryNodes('eventSearchIndex', 'power failure')
    YIELD node, score
    NOTE:Do not provide additional details other than cypher query.
    """
    return prompt

# Function to combine initial query and result for LLM
def generate_better_response(user_input, cypher_result):
    # Convert the Cypher result to a readable string
    result_summary = ""
    for record in cypher_result:
        for field in record.keys():
            value = record[field]
            result_summary += f"{field}: {value}\n"

    # Create a prompt for the LLM to improve the response
    prompt = f"""
    User asked the following question: {user_input}
    Based on the query, the following results were retrieved from the database:
    {result_summary}
    
    Please generate a response which is easy for user to understand based on user query.
    Don't mention cypher query and no need to explain what query does, just give well readable output.
    When there is warning from Neo4j Database, please return nothing.
    if the query fetch nothing don't return anything.
    """
    
    # Use OpenAI API to generate the final response
    response = client.chat.completions.create(
                model="gpt-4o",
                messages= [{"role": "user", "content": prompt}],
                max_tokens=512
            )
    final_response = response.choices[0].message.content.strip()
    return final_response

# Main function to handle user input and query generation
def handle_user_query(user_input):
    # Load ontology data from JSON
    json_data = load_json_file('RAN_ontology.json')
    
    # Create prompt and generate Cypher query
    prompt = create_prompt(user_input, json_data)
    cypher_query = generate_cypher_query(prompt)
    
    # Execute the Cypher query and get results
    cypher_result = execute_cypher_query(cypher_query)
    
    # Generate a refined response using LLM
    better_response = generate_better_response(user_input, cypher_result)
    
    print(f"\n{better_response}")




# # Example usage
# if __name__ == "__main__":
#     user_input = input("Enter your query: ")
#     handle_user_query(user_input)

