# Graph RAG Pipeline Documentation

## Dataset Description and Source

The dataset used for this project was sourced from Kaggle: Healthcare Dataset https://www.kaggle.com/datasets/prasad22/healthcare-dataset. It contains information about patients, their admissions, and the healthcare services provided. Below is a brief description of the dataset's columns:

- **Name**: Name of the patient.
- **Age**: Age of the patient at admission.
- **Gender**: Gender of the patient (Male/Female).
- **Blood Type**: Patient's blood type (e.g., A+, O-).
- **Medical Condition**: Primary medical condition or diagnosis (e.g., Diabetes, Hypertension).
- **Date of Admission**: Admission date.
- **Doctor**: Name of the doctor responsible for care.
- **Hospital**: Healthcare facility name.
- **Insurance Provider**: Insurance provider name (e.g., Aetna, Blue Cross).
- **Billing Amount**: Total billed amount for healthcare services.
- **Room Number**: Room number of the patient during admission.
- **Admission Type**: Type of admission (Emergency, Elective, Urgent).
- **Discharge Date**: Discharge date based on admission date and length of stay.
- **Medication**: Prescribed medication (e.g., Aspirin, Ibuprofen).
- **Test Results**: Results of a medical test (Normal, Abnormal, Inconclusive).

## Knowledge Graph Design and Creation Process

### Data Preparation
The dataset was preprocessed to ensure compatibility with Neo4j.

### Graph Structure
- **Nodes**: Represent entities such as patients, doctors, hospitals, medical conditions, and medications.
- **Relationships**: Connect these nodes, such as `TREATED_BY`, `HAS_CONDITION`, `ADMITTED_TO`, and `PRESCRIBED`.

### Graph Loading
The CSV file was imported into Neo4j using Cypher queries to create nodes and relationships.

### Embedding Integration
- Node embeddings were generated using BERT, extracting semantic information from node properties and relationships.
- Embeddings were stored as a property within the corresponding Neo4j nodes.

## Graph RAG Pipeline Implementation

### Embedding Generation
- Used BERT ("bert-base-uncased") to generate embeddings for query text and graph nodes.
- Combined node properties and relationship information into a single textual description for embedding generation.

### Cosine Similarity
Calculated cosine similarity between query embeddings and node embeddings to find the most relevant node(s).

### Context Retrieval
Retrieved information about the identified node and its related nodes to build context.

### Response Generation
Used the retrieved context as input for **LLaMA 3.3** LLM to generate the final response.

## Backend and Frontend Development

### Backend
- **Framework**: Flask
- **Key Features**:
  - Query embedding generation using BERT.
  - Cosine similarity calculation for node matching.
  - Neo4j interaction to fetch node data and relationships.
  - Integration with LLaMA 3.3 for generating responses.

### Frontend
- **Framework**: React
- **Key Features**:
  - User interface for querying the knowledge graph.
  - Display of retrieved context and generated answers.
  - Dynamic interaction with the backend to fetch results.

## Prompt Engineering Techniques
- Designed prompts for LLaMA 3.3 that effectively utilized the retrieved context.
- Included specific instructions to ensure accurate and context-aware responses.
- Tested and refined prompts iteratively to handle diverse query types.

## Challenges and Solutions

### Embedding Size Mismatch
- **Challenge**: Neo4j embeddings are 128-dimensional, while BERT generates 768-dimensional embeddings.
- **Solution**: Used BERT for both query and node embeddings to maintain consistency.

### Node and Relationship Representation
- **Challenge**: Encoding complex graph structures into text for embeddings.
- **Solution**: Created descriptive text combining node properties and relationships.

### Performance Optimization
- **Challenge**: Querying a large graph and generating embeddings for all nodes can be computationally expensive.
- **Solution**: Precomputed and stored embeddings for nodes in Neo4j.

## Tools and Technologies Used

- **Data Storage**: Neo4j
- **Embedding Model**: BERT ("bert-base-uncased")
- **LLM**: LLaMA 3.3
- **Backend**: Flask
- **Frontend**: React
- **Programming Languages**: Python (backend), JavaScript (frontend)

### Libraries
- **PyTorch**: For embedding generation.
- **Transformers**: For BERT model.
- **Neo4j Python Driver**: For interacting with Neo4j.

## How to Use

### Setup
1. Start the Neo4j database and load the dataset.
2. Run the backend Flask application.
3. Start the React frontend application.

### Querying
1. Enter a query in the frontend interface.
2. View the generated response and related context.


# Generative AI Full-Stack Developer Project

This project integrates a React frontend with a Flask backend, leveraging Neo4j for graph database management and Hugging Face's BERT model for NLP tasks. It also includes a Groq framework for optimization and uses cosine similarity for comparing text embeddings.

## Project Setup

### Prerequisites
- Node.js (for React)
- Python 3.x (for Flask)
- Neo4j (for graph database)
- Groq framework
- Hugging Face Transformers library

### Frontend Setup (React)
1. Install required packages for the React frontend:
    ```bash
    npm install axios
    ```

2. Run the React frontend:
    ```bash
    npm run start
    ```

### Backend Setup (Flask)
1. Install required Python packages:
    ```bash
    pip install neo4j transformers torch numpy flask flask-cors sklearn groq
    ```

2. Run the Flask backend:
    ```bash
    python back_end.py
    ```



