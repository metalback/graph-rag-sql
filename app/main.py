import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .db_connector import DatabaseConnector
from .llm_api_connector import LLMConnectorFactory
from .graph_rag import GraphRAG
from .config import settings

app = Flask(__name__, template_folder='templates')
# Enable CORS based on settings (allow all if not specified)
if settings.CORS_ORIGINS:
  CORS(app, origins=settings.CORS_ORIGINS)
else:
  CORS(app)

# Initialize components
db_connector = DatabaseConnector()
llm_connector = LLMConnectorFactory.from_env_or_config()
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

# -----------------------------
# JSON APIs
# -----------------------------
@app.route('/api/build-graph', methods=['POST'])
def api_build_graph():
  """
  Build or load the knowledge graph (and refresh DB cache if needed).
  Returns JSON status.
  """
  try:
    # Ensure DB cache is up-to-date
    db_connector.connect_and_cache()

    # Build or load the graph
    graph_rag.build_or_load_graph()

    return jsonify({"status": "ok", "message": "Graph built/loaded successfully"}), 200
  except Exception as e:
    return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/query', methods=['POST'])
def api_query():
  """
  Answer a user query using the knowledge graph context and the LLM.
  Expects JSON: {"prompt": "..."}
  Returns JSON with generated SQL (and optionally context).
  """
  try:
    data = request.get_json(silent=True) or {}
    user_prompt = data.get('prompt')
    if not user_prompt:
      return jsonify({"status": "error", "message": "Missing 'prompt' in JSON body"}), 400

    # Optionally ensure graph exists
    graph_rag.build_or_load_graph()

    # Prepare context for the LLM
    context = graph_rag.get_context(user_prompt)

    # Generate SQL using LLM
    llm = llm_connector.get_llm()
    prompt_template = PromptTemplate(
        input_variables=["context", "user_prompt"],
        template=(
          "Given the following database context and incomplete sample values:\n{context}\n\n"
          "Generate SQL for the following request:\n{user_prompt}\n\n"
          "If values do not exist in the sample data, assume that they may exist in the full database."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    sql_result = chain.run(context=context, user_prompt=user_prompt)

    return jsonify({
      "status": "ok",
      "sql": sql_result,
      "context": context
    }), 200
  except Exception as e:
    return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
