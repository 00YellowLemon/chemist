import os
import sys
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure event loop compatibility on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# Set up GCP credentials for local development or Cloud Run
from credentials_helper import setup_google_credentials
setup_google_credentials()

if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = "restaurant-c1836"

# Add current directory to path to load local modules
sys.path.append(os.path.abspath("."))
from agent import create_langchain_agent
from checkpointer import get_async_checkpointer, get_async_pool, close_async_pool
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.types import Command

# Pydantic Schemas
class ChatRequest(BaseModel):
    message: Optional[str] = Field(default=None, description="The text message input from the user.")
    image_base64: Optional[str] = Field(default=None, description="Optional base64 encoded image string (e.g. for vision tasks).")
    image_mime_type: Optional[str] = Field(default="image/jpeg", description="MIME type of the uploaded image.")
    thread_id: Optional[str] = Field(default=None, description="Thread ID to maintain session memory. A new one will be generated if not provided.")
    resume: Optional[Dict[str, Any]] = Field(default=None, description="Optional resume decision payload to recover from an interrupt.")

class MessageResponse(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "completed" or "paused"
    messages: List[MessageResponse]
    interrupts: Optional[List[Dict[str, Any]]] = None

# Lifespan Handler for DB connection pool
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Open database connection pool
    pool = get_async_pool()
    await pool.open()
    yield
    # Shutdown: Close database connection pool
    await close_async_pool()

app = FastAPI(
    title="AI Pharmacy Assistant API",
    description="Exposes endpoints for chat interaction and streaming with the Pharmacy Assistant Agent.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
origins = [
    "https://chemist-ai-ruddy.vercel.app",
    "https://chemist-ai-ruddy.vercel.app/",
    "http://localhost",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://localhost(:\d+)?|https?://127\.0\.0\.1(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def serialize_message(msg) -> Dict[str, Any]:
    """Serializes a LangChain message into a standard API role/content response format."""
    msg_dict = {
        "role": msg.type,
        "content": msg.content,
    }
    if hasattr(msg, "name") and msg.name:
        msg_dict["name"] = msg.name
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        # tool_calls can contain non-serializable elements (like tool_call object list); serialize it
        msg_dict["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
            for tc in msg.tool_calls
        ]
    return msg_dict

def get_message_chunk_text(chunk: AIMessageChunk) -> str:
    """Helper to extract clean string tokens from an AIMessageChunk."""
    content = chunk.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content) if content else ""

def format_sse(event: str, data: Any) -> str:
    """Formats event data into standard SSE protocol format."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Normal chat invocation endpoint. Returns the final response state and active interrupts (if any)."""
    thread_id = request.thread_id or f"thread-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        async with get_async_checkpointer() as checkpointer:
            agent = create_langchain_agent(checkpointer=checkpointer)

            # Determine whether we are resuming from an interrupt or starting a new turn
            if request.resume:
                inputs = Command(resume=request.resume)
            else:
                if request.image_base64:
                    text_content = request.message or "Please analyze this prescription image and extract the medication name, dosage, and quantity."
                    content = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": f"data:{request.image_mime_type};base64,{request.image_base64}"}}
                    ]
                else:
                    content = request.message or ""
                inputs = {"messages": [{"role": "user", "content": content}]}

            # Invoke agent asynchronously
            await agent.ainvoke(inputs, config=config)

            # Retrieve final execution state
            state = await agent.aget_state(config)
            
            # Extract serialized messages
            messages_out = [serialize_message(msg) for msg in state.values.get("messages", [])]

            # Detect interrupts
            status = "completed"
            interrupt_details = None
            if state.next:
                status = "paused"
                for task in state.tasks:
                    if task.interrupts:
                        interrupt_details = [
                            {"id": interrupt.id, "value": interrupt.value}
                            for interrupt in task.interrupts
                        ]
                        break

            return ChatResponse(
                thread_id=thread_id,
                status=status,
                messages=messages_out,
                interrupts=interrupt_details
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent: {str(e)}")

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events (SSE). Streams tokens, tool calls, and interrupts."""
    thread_id = request.thread_id or f"thread-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        try:
            async with get_async_checkpointer() as checkpointer:
                agent = create_langchain_agent(checkpointer=checkpointer)

                if request.resume:
                    inputs = Command(resume=request.resume)
                else:
                    if request.image_base64:
                        text_content = request.message or "Please analyze this prescription image and extract the medication name, dosage, and quantity."
                        content = [
                            {"type": "text", "text": text_content},
                            {"type": "image_url", "image_url": {"url": f"data:{request.image_mime_type};base64,{request.image_base64}"}}
                        ]
                    else:
                        content = request.message or ""
                    inputs = {"messages": [{"role": "user", "content": content}]}

                current_node = None

                # Yield events as we process messages
                async for chunk, metadata in agent.astream(inputs, config=config, stream_mode="messages"):
                    node = metadata.get("langgraph_node")
                    if node and node != current_node:
                        current_node = node
                        yield format_sse("node", {"node": current_node})

                    if isinstance(chunk, AIMessageChunk):
                        token = get_message_chunk_text(chunk)
                        if token:
                            yield format_sse("token", {"token": token})

                        if chunk.tool_calls:
                            for tc in chunk.tool_calls:
                                yield format_sse("tool_call", {
                                    "name": tc.get("name"),
                                    "args": tc.get("args"),
                                    "id": tc.get("id")
                                })

                    elif isinstance(chunk, ToolMessage):
                        yield format_sse("tool_result", {
                            "name": chunk.name,
                            "content": chunk.content,
                            "tool_call_id": chunk.tool_call_id
                        })

                # Check if the execution paused on an interrupt
                state = await agent.aget_state(config)
                if state.next:
                    interrupt_details = None
                    for task in state.tasks:
                        if task.interrupts:
                            interrupt_details = [
                                {"id": interrupt.id, "value": interrupt.value}
                                for interrupt in task.interrupts
                            ]
                            break
                    yield format_sse("interrupt", {
                        "thread_id": thread_id,
                        "interrupts": interrupt_details
                    })
                else:
                    yield format_sse("complete", {
                        "thread_id": thread_id
                    })

        except Exception as e:
            yield format_sse("error", {"detail": str(e)})

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
