from flask import Flask, request, jsonify
from neo4j import GraphDatabase
import re
from flask_cors import CORS  # Import CORS here
from groq import Groq  # Ensure your Groq client is available

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS to allow cross-origin requests
CORS(app)

# Neo4j Connector Class
class Neo4jConnector:
    def __init__(self, uri, user, password, database="neo4j"):
        """
        Initializes the Neo4j connection.

        :param uri: Neo4j connection URI (e.g., bolt://localhost:7687)
        :param user: Neo4j username
        :param password: Neo4j password
        :param database: The name of the database (default is "neo4j")
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def query(self, cypher_query, parameters=None):
        """
        Runs a Cypher query and returns the result.

        :param cypher_query: The Cypher query string to run.
        :param parameters: Optional parameters for the query (default is None).
        :return: A list of dictionaries with the query results.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, parameters)
            return [record.data() for record in result]

# Initialize Groq client (replace with your actual API key)
client = Groq(api_key="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")

# Neo4j connection (replace with your credentials)
neo4j_conn = Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")

# Helper function to extract name from user query
def extract_name(query):
    """
    Extracts the name of an entity (patient, doctor, or hospital) from the user query.

    :param query: The user's query.
    :return: Extracted name as a string.
    """
    match = re.search(r"\b(?:patient|doctor|hospital)\s+([\w\s]+)", query, re.IGNORECASE)
    return match.group(1).strip() if match else ""

# Function to fetch relevant data from Neo4j
def get_relevant_data(user_query):
    """
    Fetches relevant information from Neo4j based on the user query.
    Compares the patient/doctor/hospital names in lowercase.
    """
    extracted_name = extract_name(user_query)
    if not extracted_name:
        return []

    # Convert the extracted name from the query to lowercase for comparison
    extracted_name_lower = extracted_name.lower()

    if "patient" in user_query.lower():
        cypher_query = """
        MATCH (p:Patient)-[:SUFFERS_FROM]->(c:Condition),
              (p)-[:ADMITTED_TO]->(h:Hospital),
              (p)-[:HAS_DOCTOR]->(d:Doctor),
              (p)-[:HAS_INSURANCE]->(i:InsuranceProvider)
        WHERE toLower(p.Name) = $name
        RETURN p.Name AS Patient, 
               c.Name AS Condition, 
               h.Name AS Hospital, 
               d.Name AS Doctor, 
               i.Name AS InsuranceProvider
        """
        parameters = {"name": extracted_name_lower}
    elif "hospital" in user_query.lower():
        cypher_query = """
        MATCH (d:Doctor)-[:WORKS_AT]->(h:Hospital)
        WHERE toLower(h.Name) = $name
        RETURN d.Name AS Doctor, h.Name AS Hospital
        """
        parameters = {"name": extracted_name_lower}
    elif "doctor" in user_query.lower():
        cypher_query = """
        MATCH (d:Doctor)<-[:HAS_DOCTOR]-(p:Patient)-[:SUFFERS_FROM]->(c:Condition)
        WHERE toLower(d.Name) = $name
        RETURN d.Name AS Doctor, c.Name AS Condition
        """
        parameters = {"name": extracted_name_lower}
    else:
        return []

    return neo4j_conn.query(cypher_query, parameters)

# Function to format the data retrieved from Neo4j into a suitable format for LLM
def format_data_for_llm(data):
    """
    Formats Neo4j data into a string suitable for LLM input.

    :param data: List of query results from Neo4j.
    :return: Formatted string for the LLM.
    """
    if not data:
        return "No relevant information found in the knowledge graph."
    
    formatted = ""
    for record in data:
        formatted += ", ".join([f"{key}: {value}" for key, value in record.items()]) + "\n"
    
    return f"Relevant knowledge:\n{formatted.strip()}\n"

# Function to generate response using Groq LLM
def generate_response(user_query, context):
    """
    Uses Groq LLM to generate a response based on the user query and context.

    :param user_query: User's input query.
    :param context: Context retrieved from the knowledge graph.
    :return: The AI-generated response.
    """
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main Graph RAG pipeline function
def graph_rag_pipeline(user_query):
    """
    Executes the Graph RAG pipeline.

    :param user_query: The query from the user.
    :return: AI-generated response.
    """
    # Step 1: Retrieve relevant data
    data = get_relevant_data(user_query)
    
    # Step 2: Format data for the LLM
    context = format_data_for_llm(data)
    
    # Step 3: Generate response using Groq
    response = generate_response(user_query, context)
    
    return response

# Endpoint to handle user queries
@app.route('/query', methods=['POST'])
def query():
    """
    Endpoint to handle user queries.

    :return: JSON response containing the generated answer.
    """
    user_query = request.json.get('query')
    
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Execute the Graph RAG pipeline
    response = graph_rag_pipeline(user_query)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
