import os
import asyncio
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from checkpointer import get_sync_checkpointer, close_sync_pool, get_async_checkpointer, close_async_pool

# State schema
class State(TypedDict):
    count: int

# Simple node that increments the counter
def increment_node(state: State) -> dict:
    current = state.get("count")
    if current is None:
        current = 0
    return {"count": current + 1}

# Build a simple state graph
builder = StateGraph(State)
builder.add_node("increment", increment_node)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)


def test_sync_persistence():
    print("\n--- Testing Synchronous Persistence ---")
    with get_sync_checkpointer() as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        
        thread_id = "test-sync-thread-101"
        config = {"configurable": {"thread_id": thread_id}}
        
        # First invocation: initial value 10
        print("Invoking graph first time (input count = 10)...")
        state1 = graph.invoke({"count": 10}, config)
        print("After first invocation state count:", state1.get("count"))
        
        # Second invocation: no input (should load from DB and increment)
        print("Invoking graph second time...")
        state2 = graph.invoke({}, config)
        print("After second invocation state count:", state2.get("count"))
        
        assert state2["count"] == 12, f"Expected count to be 12, got {state2.get('count')}"
        print("Sync persistence check passed successfully!")


async def test_async_persistence():
    print("\n--- Testing Asynchronous Persistence ---")
    async with get_async_checkpointer() as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        
        thread_id = "test-async-thread-101"
        config = {"configurable": {"thread_id": thread_id}}
        
        # First invocation: initial value 20
        print("Invoking graph first time async (input count = 20)...")
        state1 = await graph.ainvoke({"count": 20}, config)
        print("After first invocation state count:", state1.get("count"))
        
        # Second invocation: no input (should load from DB and increment)
        print("Invoking graph second time async...")
        state2 = await graph.ainvoke({}, config)
        print("After second invocation state count:", state2.get("count"))
        
        assert state2["count"] == 22, f"Expected count to be 22, got {state2.get('count')}"
        print("Async persistence check passed successfully!")


async def main():
    try:
        await test_async_persistence()
    finally:
        await close_async_pool()


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        test_sync_persistence()
    finally:
        close_sync_pool()
        
    asyncio.run(main())
