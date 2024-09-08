import os
from flask import Flask, render_template, request
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from db_connector import DatabaseConnector
from llm_api_connector import LLMConnector
from graph_rag import GraphRAG

app = Flask(__name__, template_folder='templates')

# Initialize components
db_connector = DatabaseConnector()
llm_connector = LLMConnector()
graph_rag = GraphRAG()

@app.route('/', methods=['GET', 'POST'])
def index():
  """
  Render the main page and process user prompts.
  """
  if request.method == 'POST':
      user_prompt = request.form['prompt']
      
      # Connect to the database and update cache if necessary
      db_connector.connect_and_cache()
      
      # Build or load the graph RAG system
      graph_rag.build_or_load_graph()
      
      # Prepare context for the LLM
      context = graph_rag.get_context(user_prompt)
      
      # Generate SQL using LLM
      llm = llm_connector.get_llm()
      prompt_template = PromptTemplate(
          input_variables=["context", "user_prompt"],
          template="Given the following database context and incomplete sample values:\n{context}\n\nGenerate SQL for the following request:\n{user_prompt}\n\n If values do not exist in the sample data, assume that they may exist in the full database."
      )
      chain = LLMChain(llm=llm, prompt=prompt_template)
      sql_result = chain.run(context=context, user_prompt=user_prompt)
      
      return render_template('user_interface.html', result=sql_result)
  
  return render_template('user_interface.html')

if __name__ == '__main__':
  app.run(debug=True)

# Created/Modified files during execution:
print("No files were created or modified during the execution of main.py")