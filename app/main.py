import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from .database import get_database_connector
from .graph import GraphRAG
from .config import settings
from .llm.gemini import GeminiLLM
import pandas as pd
try:
  from .llm.openai import OpenAILLM  # type: ignore
except Exception:  # pragma: no cover
  OpenAILLM = None  # type: ignore
try:
  from .llm.anthropic import AnthropicLLM  # type: ignore
except Exception:  # pragma: no cover
  AnthropicLLM = None  # type: ignore

app = Flask(__name__, template_folder='templates')
# Enable CORS based on settings (allow all if not specified)
if settings.CORS_ORIGINS:
  CORS(app, origins=settings.CORS_ORIGINS)
else:
  CORS(app)

# Initialize components
db_connector = get_database_connector()  # selects provider via env (DB_PROVIDER), connects immediately
graph_rag = GraphRAG()

# Database run_sql bound from the selected connector
run_sql = db_connector.run_sql  # type: ignore

def _create_llm():
  provider = (settings.LLM_PROVIDER or os.environ.get('LLM_PROVIDER', 'google')).lower()
  if provider == 'google':
    return GeminiLLM()
  if provider == 'openai':
    if OpenAILLM is None:
      raise ImportError('langchain-openai no instalado; no se puede usar OpenAI')
    return OpenAILLM()  # type: ignore
  if provider == 'anthropic':
    if AnthropicLLM is None:
      raise ImportError('langchain-anthropic no instalado; no se puede usar Anthropic')
    return AnthropicLLM()  # type: ignore
  raise ValueError(f"Proveedor LLM no soportado: {provider}")

llm = _create_llm()

@app.route('/', methods=['GET', 'POST'])
def index():
  """
  Render the main page and process user prompts.
  """
  if request.method == 'POST':
      user_prompt = request.form['prompt']
      
      # Connect to the database and update cache if necessary
      if hasattr(db_connector, 'connect_and_cache'):
        db_connector.connect_and_cache()
      
      # Build or load the graph RAG system
      graph_rag.build_or_load_graph()
      
      # Prepare context for the LLM
      context = graph_rag.get_context(user_prompt)
      
      # Generate SQL using new LLM interface
      prompt = (
          "Given the following database context and incomplete sample values:\n"
          f"{context}\n\n"
          "Generate SQL for the following request:\n"
          f"{user_prompt}\n\n"
          "If values do not exist in the sample data, assume that they may exist in the full database."
      )
      sql_result = llm.submit_prompt(prompt)

      # Try executing the SQL and prepare a preview of rows
      exec_error = None
      preview_rows = []
      columns = []
      try:
        df = run_sql(sql_result)
        columns = list(df.columns)
        preview_rows = df.head(50).astype(str).values.tolist()
      except Exception as ex:
        exec_error = str(ex)
      
      return render_template('user_interface.html', result=sql_result, columns=columns, rows=preview_rows, exec_error=exec_error)
  
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
    if hasattr(db_connector, 'connect_and_cache'):
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

    # Generate SQL using new LLM interface
    prompt = (
        "Given the following database context and incomplete sample values:\n"
        f"{context}\n\n"
        "Generate SQL for the following request:\n"
        f"{user_prompt}\n\n"
        "If values do not exist in the sample data, assume that they may exist in the full database."
    )
    sql_result = llm.submit_prompt(prompt)

    # Execute SQL and return results if possible
    resp = {"status": "ok", "sql": sql_result, "context": context}
    try:
      df = run_sql(sql_result)
      resp["columns"] = list(df.columns)
      resp["rows"] = df.head(200).astype(str).values.tolist()
    except Exception as ex:
      resp["exec_error"] = str(ex)

    return jsonify(resp), 200
  except Exception as e:
    return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
