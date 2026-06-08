import os
import sys
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

# Load variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def setup_database():
    if not DATABASE_URL:
        print("Error: DATABASE_URL not set in .env file", file=sys.stderr)
        sys.exit(1)
        
    print("Connecting to the database and setting up checkpointer tables...")
    try:
        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            checkpointer.setup()
            print("Successfully initialized the PostgreSQL checkpointer schema!")
    except Exception as e:
        print(f"Failed to setup checkpointer: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    setup_database()
