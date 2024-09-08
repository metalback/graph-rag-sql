import os
import json
import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

class GraphRAG:
  def __init__(self):
      self.vector_store = None
      self.graph = None
      self.cache_dir = 'cache'
      self.vector_store_path = 'cache/vector_store.faiss'
      self.graph_path = 'cache/graph.gml'

  def build_or_load_graph(self):
      if os.path.exists(self.vector_store_path) and os.path.exists(self.graph_path):
          self.load_graph()
      else:
          self.build_graph()

  def build_graph(self):
      documents = []
      self.graph = nx.Graph()

      print(f"Checking cache directory: {self.cache_dir}")
      if not os.path.exists(self.cache_dir):
          print(f"Cache directory does not exist. Creating it.")
          os.makedirs(self.cache_dir)

      # Load all documents into the vector database
      for file in os.listdir(self.cache_dir):
          if file.endswith('.json'):
              print(f"Processing file: {file}")
              db_name, table_name = file.split('_')
              table_name = table_name.split('.')[0]
              
              file_path = os.path.join(self.cache_dir, file)
              try:
                  with open(file_path, 'r') as f:
                      content = json.load(f)
                      if content:
                          # Extract column names (first level of the JSON file)
                          column_names = list(content.keys()) if content else []
                          # Convert content back to string for vector store
                          content_str = json.dumps(content)
                          
                          documents.append(Document(
                              page_content=content_str,
                              metadata={
                                  'db': db_name,
                                  'table': table_name,
                                  'columns': column_names
                              }
                          ))
                          print(f"Added document for {db_name}.{table_name} with columns: {column_names}")
                      else:
                          print(f"Warning: {file} is empty")
              except Exception as e:
                  print(f"Error reading {file}: {str(e)}")

      print(f"Total documents processed: {len(documents)}")

      if not documents:
          print("No documents were processed. Cannot create vector store.")
          return

      # Create vector store
      try:
          embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.environ['GOOGLE_AI_KEY'], model="models/embedding-001")
          self.vector_store = FAISS.from_documents(documents, embeddings)
          print("Vector store created successfully")
      except Exception as e:
          print(f"Error creating vector store: {str(e)}")
          return

      # Build the knowledge graph based on similarities
      similarity_threshold = 0.3  # Adjust this threshold as needed
      for i, doc1 in enumerate(documents):
          node1 = f"{doc1.metadata['db']}.{doc1.metadata['table']}"
          self.graph.add_node(node1, columns=doc1.metadata['columns'])
          for j, doc2 in enumerate(documents[i+1:], start=i+1):
              node2 = f"{doc2.metadata['db']}.{doc2.metadata['table']}"
              similarity = self.vector_store.similarity_search_with_score(doc1.page_content, k=1)[0][1]
              if similarity >= similarity_threshold:
                  self.graph.add_edge(node1, node2, weight=similarity)

      # Save vector store and graph
      try:
          self.vector_store.save_local(self.vector_store_path)
          nx.write_gml(self.graph, self.graph_path)
          print("Vector store and graph saved successfully")
      except Exception as e:
          print(f"Error saving vector store or graph: {str(e)}")

  def load_graph(self):
      try:
          embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.environ['GOOGLE_AI_KEY'], model="models/embedding-001")
          self.vector_store = FAISS.load_local(self.vector_store_path, embeddings)
          self.graph = nx.read_gml(self.graph_path)
          print("Vector store and graph loaded successfully")
      except Exception as e:
          print(f"Error loading vector store or graph: {str(e)}")

  def get_context(self, query):
      if not self.vector_store:
          return "Error: Vector store not initialized."

      # Perform similarity search
      relevant_docs = self.vector_store.similarity_search(query, k=3)
      
      context = "Database structure:\n"
      for doc in relevant_docs:
          db_name = doc.metadata['db']
          table_name = doc.metadata['table']
          node = f"{db_name}.{table_name}"
          
          # Add information about the table
          context += f"- Table: {node}\n"
          context += f"  Columns: {', '.join(doc.metadata['columns'])}\n"
          context += f"  Sample data: {doc.page_content}\n"
          
          # Add information about related tables (connected nodes)
          related_tables = list(self.graph.neighbors(node))
          if related_tables:
              context += f"  Related tables: {', '.join(related_tables)}\n"
              
              # Include information from connected nodes
              for related_table in related_tables:
                  related_columns = self.graph.nodes[related_table]['columns']
                  context += f"    - {related_table} columns: {', '.join(related_columns)}\n"
                  context += f"    - {related_table} Sample data: {doc.page_content}\n"
          
          context += "\n"
      
      return context