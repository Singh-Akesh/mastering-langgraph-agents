import time
import uuid
import json
import hashlib
from typing import List, Optional, Dict, Any, Generator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
from rag_pipeline import rag_query as real_rag_query

# ------------------------------
# === CONFIG / MODELS SETUP ===
# ------------------------------
# Control which models/endpoints you want exposed here.
MODELS = {
    "rag-model": {
        "id": "rag-model",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "rag-backend",
        "description": "RAG-backed chat model (uses internal rag_query)",
        "type": "chat"
    },
    "rag-complete": {
        "id": "rag-complete",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "rag-backend",
        "description": "RAG-backed text completion",
        "type": "completion"
    },
    "rag-embed": {
        "id": "rag-embed",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "rag-backend",
        "description": "Embedding model backed by your embedding function",
        "type": "embedding"
    }
}

# ------------------------------
# === Replace / Plug Points ===
# ------------------------------
# Replace this with your real RAG pipeline function.
# It receives the latest user query and the full chat history (list of {"role","content"}).
def rag_query(query: str, history: List[Dict[str, str]]) -> str:
    """
    # >>> Replace with your pipeline. Should return assistant text.
    Example contract:
      - query: latest user text
      - history: entire conversation as list of {"role": "...", "content": "..."}
    """
    # Example placeholder: echo + length of history
    out  = real_rag_query(query)
    return out["answer"]
    # return f"RAG response to: '{query}' (history_len={len(history)})"

# Replace this with your real embedding function.
def embed_text(text: str) -> List[float]:
    """
    Return a deterministic pseudo-embedding (list[float]) for the input text.
    Replace with your model/vector store embedding call.
    """
    # Simple deterministic hash-based vector (not useful for real downstream)
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = [(b / 255.0) for b in h[:64]]  # 64-d vector of floats in [0,1]
    return vec

# ------------------------------
# === Session / Storage ===
# ------------------------------
# In-memory sessions. If you want persistence, replace with a DB or file-backed store.
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in chat_sessions:
        return session_id
    if session_id and session_id not in chat_sessions:
        # create empty session with provided id
        chat_sessions[session_id] = []
        return session_id
    new_id = str(uuid.uuid4())
    chat_sessions[new_id] = []
    return new_id

# ------------------------------
# === FastAPI / Schemas ===
# ------------------------------
app = FastAPI(title="RAG OpenAI-Compatible API (drop-in for Open WebUI)")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    session_id: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False
    session_id: Optional[str] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]  # OpenAI allows single string or list. We accept list.

# ------------------------------
# === Utilities ===
# ------------------------------
def openai_usage_stub(prompt_tokens=0, completion_tokens=0) -> Dict[str,int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }

# SSE/event helper for streaming text tokens
async def sse_event_stream_text(generator: Generator[str, None, None]):
    """
    Given a text generator that yields chunks (tokens), format SSE events as OpenAI does:
    sends lines like: data: {"id": "...", "object": "chat.completion.chunk", ...}
    and finishes with data: [DONE]
    """
    # yield initial meta? Not necessary. We'll send incremental chunks.
    try:
        async for chunk in async_iter_from_generator(generator):
            data = {
                "choices": [
                    {
                        "delta": {"content": chunk},
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(data)}\n\n"
        # finish
        yield "data: [DONE]\n\n"
    except Exception as e:
        # If something goes wrong, signal end
        yield f"data: [DONE]\n\n"

async def async_iter_from_generator(gen: Generator[str, None, None]):
    # Wrap a sync generator into async iterator (run in threadpool)
    loop = asyncio.get_event_loop()
    for chunk in gen:
        yield chunk
        await asyncio.sleep(0)  # let event loop breathe

# Simple generator for streaming tokens from rag_query.
# Replace with your token-by-token iterator if you have one.
def rag_streaming_generator(query: str, history: List[Dict[str,str]]):
    text = rag_query(query, history)
    # naive tokenization by words — replace with proper token streaming if available
    for token in text.split():
        yield token + " "
    # completed

# ------------------------------
# === ENDPOINTS ===
# ------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": list(MODELS.values())}

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    model = MODELS.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="model_not_found")
    return model

# Chat completions (OpenAI Chat API compatible)
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # validate model
    if req.model not in MODELS:
        raise HTTPException(status_code=404, detail="model_not_found")

    # get session
    session_id = get_or_create_session(req.session_id)
    history = chat_sessions.get(session_id, [])

    # append incoming messages to history (we store full messages)
    for m in req.messages:
        history.append({"role": m.role, "content": m.content})

    # find last user message (OpenAI clients typically send the user msg as last)
    last_user_msgs = [m for m in req.messages if m.role == "user"]
    if not last_user_msgs:
        raise HTTPException(status_code=400, detail="no_user_message_found")
    last_user = last_user_msgs[-1].content

    # If streaming requested, stream tokens as SSE
    if req.stream:
        # Create a simple sync generator (can be replaced by a real async generator)
        gen = rag_streaming_generator(last_user, history)

        # Append final assistant full response to session after generation
        # To avoid blocking, accumulate chunks and then append after streaming done.
        # Here we'll collect in-memory for simplicity.
        def post_stream_and_accumulate():
            text = rag_query(last_user, history)
            history.append({"role": "assistant", "content": text})
            chat_sessions[session_id] = history
            return

        # Build a wrapper generator that yields tokens and at end calls post_stream
        def wrapper_gen():
            for chunk in gen:
                yield chunk
            # ensure final append (the generator approach above already calls rag_query once, so for simplicity do again)
            post_stream_and_accumulate()

        return StreamingResponse(
            sse_event_stream_text(wrapper_gen()),
            media_type="text/event-stream"
        )

    # Non-streaming: single response
    assistant_text = rag_query(last_user, history)
    assistant_msg = {"role": "assistant", "content": assistant_text}
    history.append(assistant_msg)
    chat_sessions[session_id] = history

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": assistant_msg,
                "finish_reason": "stop"
            }
        ],
        "usage": openai_usage_stub(prompt_tokens=0, completion_tokens=0),
        "session_id": session_id
    }
    return JSONResponse(response)

# Classic completions endpoint (text completions)
@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=404, detail="model_not_found")

    session_id = get_or_create_session(req.session_id)
    history = chat_sessions.get(session_id, [])

    prompt = req.prompt or ""
    # For compatibility with completions API, add prompt as a "user" message into history
    history.append({"role": "user", "content": prompt})

    if req.stream:
        # stream via SSE similar to chat
        gen = rag_streaming_generator(prompt, history)

        def wrapper_gen():
            for chunk in gen:
                yield chunk
            # append final full text
            full = rag_query(prompt, history)
            history.append({"role": "assistant", "content": full})
            chat_sessions[session_id] = history

        async def sse_wrapper():
            async for token in async_iter_from_generator(wrapper_gen()):
                data = {
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion.chunk",
                    "choices": [
                        {"text": token, "index": 0, "finish_reason": None}
                    ]
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_wrapper(), media_type="text/event-stream")

    # non-streaming
    text = rag_query(prompt, history)
    history.append({"role": "assistant", "content": text})
    chat_sessions[session_id] = history

    resp = {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"text": text, "index": 0, "finish_reason": "stop"}
        ],
        "usage": openai_usage_stub()
    }
    return JSONResponse(resp)

# Embeddings endpoint
@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=404, detail="model_not_found")

    # Accept list of inputs and return list of vectors
    inputs = req.input
    if not isinstance(inputs, list):
        raise HTTPException(status_code=400, detail="input_must_be_list")

    data = []
    for i, text in enumerate(inputs):
        vec = embed_text(text)
        data.append({
            "object": "embedding",
            "embedding": vec,
            "index": i
        })

    resp = {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": openai_usage_stub()
    }
    return JSONResponse(resp)

# ------------------------------
# === Helper: clear sessions (dev) ===
# ------------------------------
@app.post("/dev/clear_sessions")
async def clear_sessions():
    chat_sessions.clear()
    return {"status": "cleared"}

# ------------------------------
# === Run hint (uvicorn) ===
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
