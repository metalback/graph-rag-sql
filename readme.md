# No schema? No data dictionary? No problem!

This is a generalizable graph-RAG LLM tool originally developed for EHR (electronic health records) where often there is no reliable documentation, but can be used for any large/messy/poorly documented database and not limited to healthcare data. 

Given access to a database and a prompt, it returns a valid SQL query based on the learned structure and examples from all available tables and databases.

## Motivation

Large EHR databases often have hundreds and thousands of tables and lack comprehensive documentation, since the schema is often proprietary and implemented slightly differently in different organizations, making it challenging for users to write complex SQL queries. This tool bridges that gap by automatically exploring the database structure and generating SQL code based on natural language prompts.

<p align="center">
<img width="825" alt="graphRAG" src="https://github.com/user-attachments/assets/7b48af46-01cc-47f5-b4a0-b6fec3387055">
</p>


## Components

1. `app/main.py`: Flask app that integrates LLM, GraphRAG, and database execution (run_sql).
2. `app/db_connector.py`: Legacy cache builder for sample values. Database-specific connectors live under `app/database/`.
3. `app/database/mssql/connector.py`: MSSQL connector using SQLAlchemy/pyodbc. Provides `run_sql(sql) -> pd.DataFrame`.
4. `app/graph/graph_rag.py`: Graph RAG implementation and context retrieval.
5. `app/llm/`: Modular LLM interface and providers:
    - `app/llm/base.py`: BaseLLM interface
    - `app/llm/gemini`, `app/llm/openai`, `app/llm/anthropic`: Provider implementations
6. `templates/user_interface.html`: Simple UI to input prompts and view generated SQL and results preview.

## How It Works

1. The system connects to the specified databases and caches the first X rows of each table, and detects frequent values.
2. The cached data is used to build a vector store using FAISS for efficient similarity search.
3. A knowledge graph representation of the vectore store chunks, identifying potential relationships between tables. Traversing edges and adding context.<br>
   When I originally implemented vector-store RAG without a graph, it worked for basic queries but failed to suggest joins that involve more than 2 tables and require an understanding of the relationships between tables and databases. As you can see in the above example, it can now easily join more than 2 tables.
4. When a user submits a prompt, the system: <br>
      a. Retrieves relevant context from the vector store. <br>
      b. For each table in the context, use its graph edges to load context about related tables.  <br>
      c. Sends the context-enriched prompt to the LLM.<br>
      d. Generates SQL code based on the LLM's response.<br>
6. The generated SQL code is displayed to the user through the web interface.

## Setup and Usage

1. Install the required dependencies: `pip install -r requirements.txt`
2. Set up your LLM API. For Google Generative AI Studio (free), ensure the `GOOGLE_AI_KEY` environment variable is set.
3. Run `main.py` to start the Flask server.
4. Access the web interface at `http://localhost:5000` and enter your prompts to generate SQL code.

## Note

This project is designed for demonstration purposes and uses sample SQLite databases. For production use, you should adapt the `db_connector.py` to work with your specific database (Spark,Hadoop,cloud etc.) systems and implement proper security measures.
