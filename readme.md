# No schema? No data dictionary? No problem!

This is a generalizable graph-RAG LLM tool originally developed for EHR (electronic health records) where often there is no reliable documentation, but can be used for any large/messy/poorly documented database and not limited to healthcare data. 

Given access to a database and a prompt, it returns a valid SQL query based on the learned structure and examples from all available tables and databases.

## Motivation

Large EHR databases often have hundreds and thousands of tables and lack comprehensive documentation, since the schema is often proprietary and implemented slightly differently in different organizations, making it challenging for users to write efficient SQL queries. This tool bridges that gap by automatically exploring the database structure and generating SQL code based on natural language prompts.

<p align="center">
   <img width="853" alt="graphRAG" src="https://github.com/user-attachments/assets/089b2c5e-4f43-4d8f-a471-e5960dea8620">
</p>

## Components

1. `main.py`: The main application file that integrates all components and handles user requests.
2. `db_connector.py`: Manages database connections and caching. Example given with sqlite3, set up your Spark/Hadoop/etc. connector here. 
3. `llm_api_connector.py`: Handles the connection to an LLM API. The example uses Google Generative AI studio due to their free API. If used in a healthcare setting, you need to use a PHI API approved by your institution. 
4. `graph_rag.py`: Implements the Graph RAG system using FAISS for vector storage and NetworkX for graph representation.
5. `user_interface.html`: Provides a simple interface for users to input prompts and view results while experimenting with the logic.

## How It Works

1. The system connects to the specified databases and caches the first X rows of each table, and detects frequent values.
2. The cached data is used to build a vector store using FAISS for efficient similarity search.
3. A knowledge graph representation of the vectore store chunks, identifying potential relationships between tables. When I originally implemented vector-store RAG without a graph, it worked for basic queries but failed to do complicated joins that require an understanding of the relationships between tables. 
4. When a user submits a prompt, the system:
   a. Retrieves relevant context from the vector store.
   b. For each table in the context, use its graph edges to load context about related tables.  
   c. Sends the context-enriched prompt to the LLM.
   d. Generates SQL code based on the LLM's response.
5. The generated SQL code is displayed to the user through the web interface.

## Setup and Usage

1. Install the required dependencies: `pip install -r requirements.txt`
2. Set up your LLM API. For Google Generative AI Studio (free), ensure the `GOOGLE_AI_KEY` environment variable is set.
3. Run `main.py` to start the Flask server.
4. Access the web interface at `http://localhost:5000` and enter your prompts to generate SQL code.

## Note

This project is designed for demonstration purposes and uses sample SQLite databases. For production use, you should adapt the `db_connector.py` to work with your specific database (Spark,Hadoop,cloud etc.) systems and implement proper security measures.
