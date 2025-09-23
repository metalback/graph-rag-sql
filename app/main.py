import os
import logging
import io
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, Form, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from .database import get_database_connector
from .graph import GraphRAG
from .config import settings
from .llm import create_llm_from_env
from .vector_store.document_vectors import (
  build_vectorstore,
  update_vectorstore,
  delete_vectorstore,
  get_retriever,
  get_RAG_answer,
)
import pandas as pd

# Create FastAPI app
app = FastAPI(title="Graph RAG SQL", version="1.0.0")

# Basic logging config (can override with LOG_LEVEL env)
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Configure CORS
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Templates
templates = Jinja2Templates(directory="app/templates")

# Initialize components
db_connector = get_database_connector()  # selects provider via env (DB_PROVIDER), connects immediately
graph_rag = GraphRAG()

# Database run_sql bound from the selected connector
run_sql = db_connector.run_sql  # type: ignore

llm = create_llm_from_env()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    prompt: str

class VectorQueryRequest(BaseModel):
    query: str
    k: Optional[int] = 10

class QueryResponse(BaseModel):
    status: str
    sql: Optional[str] = None
    context: Optional[str] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[List[str]]] = None
    exec_error: Optional[str] = None

class VectorQueryResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    context: Optional[str] = None
    message: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    deleted: Optional[int] = None
    chunks: Optional[int] = None
    path: Optional[str] = None
    cache_json_before: Optional[int] = None
    cache_json_after: Optional[int] = None
    created_or_existing: Optional[int] = None

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("user_interface.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def post_index(request: Request, prompt: str = Form(...)):
    """
    Process user prompts and render results.
    """
    user_prompt = prompt
    
    # Connect to the database and update cache if necessary
    if hasattr(db_connector, 'connect_and_cache'):
        db_connector.connect_and_cache()
    
    # Build or load the graph RAG system
    graph_rag.build_or_load_graph()
    
    # Prepare context for the LLM
    context = graph_rag.get_context(user_prompt)
    
    # Generate SQL using new LLM interface
    llm_prompt = (
        "Given the following database context and incomplete sample values:\n"
        f"{context}\n\n"
        "Generate SQL for the following request:\n"
        f"{user_prompt}\n\n"
        "If values do not exist in the sample data, assume that they may exist in the full database."
    )
    sql_result = llm.submit_prompt(llm_prompt)

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
    
    return templates.TemplateResponse(
        "user_interface.html", 
        {
            "request": request, 
            "result": sql_result, 
            "columns": columns, 
            "rows": preview_rows, 
            "exec_error": exec_error
        }
    )

# -----------------------------
# JSON APIs
# -----------------------------
@app.post('/api/build-graph', response_model=StatusResponse)
async def api_build_graph():
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

        return StatusResponse(status="ok", message="Graph built/loaded successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/vector/query', response_model=VectorQueryResponse)
async def api_vector_query(query_request: VectorQueryRequest):
    """
    Query the persisted vector store to retrieve similar documents and generate an answer.
    """
    try:
        user_query = query_request.query.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Missing 'query' in request body")
        
        k = query_request.k
        logger.info("/api/vector/query: query_len=%s k=%s", len(user_query), k)
        retriever = get_retriever(k=int(k))
        result = get_RAG_answer(user_query, llm, retriever)
        logger.info("/api/vector/query: answer_len=%s context_len=%s", len(result.get('answer','')), len(result.get('context','')))
        
        return VectorQueryResponse(
            status="ok",
            answer=result.get('answer'),
            context=result.get('context')
        )
    except FileNotFoundError as e:
        logger.warning("/api/vector/query: vector store not found: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("/api/vector/query: failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/graph/image')
async def api_graph_image():
    """
    Render the current knowledge graph as a PNG image.
    """
    try:
        # Ensure graph is available
        graph_rag.build_or_load_graph()
        G = graph_rag.graph
        if G is None or G.number_of_nodes() == 0:
            raise HTTPException(status_code=400, detail="Graph is empty or not built")

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=1, font_size=10, ax=ax)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/graph/delete-cache', response_model=StatusResponse)
async def api_graph_delete_cache():
    try:
        deleted = graph_rag.delete_cache()
        return StatusResponse(status="ok", deleted=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/graph/delete-db', response_model=StatusResponse)
async def api_graph_delete_db():
    try:
        deleted = graph_rag.delete_graph_db()
        return StatusResponse(status="ok", deleted=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/graph/update-db', response_model=StatusResponse)
async def api_graph_update_db():
    try:
        graph_rag.update_graph_db()
        return StatusResponse(status="ok", message="Graph DB rebuilt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/graph/update-cache', response_model=StatusResponse)
async def api_graph_update_cache():
    try:
        import os
        before = 0
        try:
            before = len([f for f in os.listdir(graph_rag.cache_dir) if f.endswith('.json')])
        except Exception:
            pass
        graph_rag.update_graph_cache(db_connector)
        after = before
        try:
            after = len([f for f in os.listdir(graph_rag.cache_dir) if f.endswith('.json')])
        except Exception:
            pass
        return StatusResponse(
            status="ok", 
            message="Cache refreshed", 
            cache_json_before=before, 
            cache_json_after=after, 
            created_or_existing=max(0, after)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/vector/build', response_model=StatusResponse)
async def api_vector_build():
    """
    Build a new Chroma vector store from documents/ (replaces existing).
    Returns: {status, chunks, path}
    """
    try:
        logger.info("/api/vector/build: starting build from documents/")
        chunks, path = build_vectorstore(documents_dir='documents')
        logger.info("/api/vector/build: completed with chunks=%s path=%s", chunks, path)
        return StatusResponse(status="ok", chunks=chunks, path=path)
    except Exception as e:
        logger.exception("/api/vector/build: failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/vector/update', response_model=StatusResponse)
async def api_vector_update():
    """
    Update (rebuild) the Chroma vector store from current documents/.
    Returns: {status, chunks, path}
    """
    try:
        logger.info("/api/vector/update: starting update from documents/")
        chunks, path = update_vectorstore(documents_dir='documents')
        logger.info("/api/vector/update: completed with chunks=%s path=%s", chunks, path)
        return StatusResponse(status="ok", chunks=chunks, path=path)
    except Exception as e:
        logger.exception("/api/vector/update: failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/vector/delete', response_model=StatusResponse)
async def api_vector_delete():
    """
    Delete the persisted Chroma vector store from disk.
    Returns: {status, deleted}
    """
    try:
        logger.info("/api/vector/delete: deleting vector store")
        deleted = delete_vectorstore()
        logger.info("/api/vector/delete: deleted=%s", deleted)
        return StatusResponse(status="ok", deleted=deleted)
    except Exception as e:
        logger.exception("/api/vector/delete: failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/query', response_model=QueryResponse)
async def api_query(query_request: QueryRequest):
    """
    Answer a user query using the knowledge graph context and the LLM.
    Returns JSON with generated SQL (and optionally context).
    """
    try:
        user_prompt = query_request.prompt
        if not user_prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt' in request body")

        # Optionally ensure graph exists
        graph_rag.build_or_load_graph()

        # Prepare context for the LLM
        context = graph_rag.get_context(user_prompt)

        # Generate SQL using new LLM interface
        llm_prompt = (
            "Given the following database context and incomplete sample values:\n"
            f"{context}\n\n"
            "Generate SQL for the following request:\n"
            f"{user_prompt}\n\n"
            "If values do not exist in the sample data, assume that they may exist in the full database."
        )
        sql_result = llm.submit_prompt(llm_prompt)

        # Execute SQL and return results if possible
        response_data = QueryResponse(status="ok", sql=sql_result, context=context)
        try:
            df = run_sql(sql_result)
            response_data.columns = list(df.columns)
            response_data.rows = df.head(200).astype(str).values.tolist()
        except Exception as ex:
            response_data.exec_error = str(ex)

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
