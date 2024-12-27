from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import groq
from flask_cors import CORS


app = Flask(__name__)

CORS(app)
uri = "bolt://localhost:7687"  
username = "neo4j"  
password = "YOUR_PASSWORD"  
database = "test2"  

driver = GraphDatabase.driver(uri, auth=(username, password))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  
    return embeddings

def get_all_node_embeddings():
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (n)-[r]->(m)
            WHERE n.embedding IS NOT NULL 
            RETURN id(n) as node_id, n.embedding as embedding, n.name as name, 
                   n.properties as node_properties, n.type AS node_type,
                   r.type AS relationship_type, 
                   m.name AS connected_node_name, m.labels AS connected_node_labels
        """)
        
        node_embeddings = []
        node_ids = []
        node_names = []
        node_types = []
        node_properties = []
        relationships_info = []
        connected_node_names = []
        connected_node_labels = []  
        
        for record in result:
            node_ids.append(record["node_id"])
            node_embeddings.append(record["embedding"])  
            node_names.append(record["name"])
            node_types.append(record["node_type"])
            node_properties.append(record["node_properties"])
            relationships_info.append(record["relationship_type"])
            connected_node_names.append(record["connected_node_name"])
            connected_node_labels.append(record["connected_node_labels"])  
        
        return node_ids, np.array(node_embeddings), node_names, node_types, node_properties, relationships_info, connected_node_names, connected_node_labels

def find_similar_nodes(query_text, top_n=5):
    # Generate the query embedding
    query_embedding = generate_query_embedding(query_text)
    
    # Get all node embeddings and names from Neo4j
    node_ids, node_embeddings, node_names, node_types, node_properties, relationships_info, connected_node_names, connected_node_labels = get_all_node_embeddings()
    
    # Calculate cosine similarities between the query embedding and all node embeddings
    similarities = cosine_similarity([query_embedding], node_embeddings)[0]
    
    # Get top N most similar nodes
    top_indices = similarities.argsort()[-top_n:][::-1]
    similar_nodes = []
    for i in top_indices:
        similar_nodes.append({
            "node_id": node_ids[i],
            "node_name": node_names[i],
            "node_type": node_types[i],
            "similarity_score": similarities[i],
            "node_properties": node_properties[i],
            "relationships": relationships_info[i],
            "connected_node_name": connected_node_names[i],
            "connected_node_labels": connected_node_labels[i], 
        })
    
    return similar_nodes

def get_node_details_by_id(node_id):
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (n)-[r]->(m)
            WHERE id(n) = $node_id
            RETURN r, m.name AS connected_node_name, m.labels AS connected_node_labels
        """, node_id=node_id)
        
        relationships = []
        connected_nodes = []
        for record in result:
            relationships.append({
                "type": record["r"].type, 
                "connected_node_name": record["connected_node_name"],
                "connected_node_labels": record["connected_node_labels"]
            })
        
        return relationships

def generate_query_embedding(query_text):
    query_embedding = get_bert_embedding(query_text)
    return query_embedding.squeeze().detach().numpy()

def print_similar_nodes(query_text, top_n=5):
    similar_nodes = find_similar_nodes(query_text, top_n)
    

def query_llm_with_node_context(query_text, node_info, relationships):
    """
    Generate a detailed context for the LLM, including both the query and the node description,
    then query the LLM for an answer based on the provided context.

    Parameters:
        query_text (str): The user's query text.
        node_info (dict): Information about the node.
            - node_name: Name of the node.
            - node_type: Type of the node (optional).
            - node_properties: Additional properties of the node (optional).
        relationships (list of dict): Information about relationships connected to the node.
            Each dict contains:
            - type: Type of relationship.
            - connected_node_name: Name of the connected node.

    Returns:
        str: The response from the LLM answering the query based on the context.
    """
    # Create a conversational node description
    node_description = (
        f"Here’s what I found about the node:\n\n"
        f"- **Name**: {node_info.get('node_name', 'Unknown')}\n"
        f"- **Type**: {node_info.get('node_type', 'Not specified')}\n"
        f"- **Properties**: {node_info.get('node_properties', 'No additional properties listed')}\n\n"
        f"It’s connected to the following nodes through these relationships:\n"
    )

    if relationships:
        for relationship in relationships:
            node_description += (
                f"  • Relationship Type: {relationship['type']}\n"
                f"    Connected Node: {relationship['connected_node_name']}\n"
            )
    else:
        node_description += "  (No relationships found)\n"

    
    # Inform the LLM that information has been gathered using a Neo4j graph
    prompt = (
        f"Query: {query_text}\n\n"
        f"I have used a Neo4j graph database to find relevant information related to this query.\n"
        f"Here’s the information I found about the relevant node:\n\n"
        f"{node_description}\n"
        f"please answer the query using the provided context."
    )

    # Initialize Groq client
    client = groq.Client(api_key="YOUR_API_KEY")

    # Query the LLM for an answer based on the query and node context
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    
    similar_nodes = find_similar_nodes(query_text, top_n=1)
    if similar_nodes:
        node_info = similar_nodes[0]
        relationships = get_node_details_by_id(node_info['node_id'])
        llm_response = query_llm_with_node_context(query_text, node_info, relationships)
        return jsonify({"response": llm_response})
    else:
        return jsonify({"error": "No similar nodes found."}), 404


if __name__ == "__main__":
    app.run(port=5002)
