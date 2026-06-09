import os
import sys
import json
import base64
from typing import List, Dict, Any
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware

# Load environment variables from .env
load_dotenv()

def get_credentials():
    """Load service account credentials if configured, otherwise returns None to use Application Default Credentials (ADC)."""
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        print("[INFO] Loading service account credentials from GOOGLE_SERVICE_ACCOUNT_JSON environment variable.")
        try:
            from google.oauth2 import service_account
            info = json.loads(sa_json)
            return service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except Exception as e:
            print(f"[ERROR] Failed to load credentials from GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
            return None

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        print("[INFO] GOOGLE_APPLICATION_CREDENTIALS not set in environment. Using default ADC.")
        return None

    # Check if GOOGLE_APPLICATION_CREDENTIALS contains the actual JSON string
    if cred_path.strip().startswith("{"):
        print("[INFO] GOOGLE_APPLICATION_CREDENTIALS contains inline JSON. Loading directly.")
        try:
            from google.oauth2 import service_account
            info = json.loads(cred_path)
            return service_account.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse GOOGLE_APPLICATION_CREDENTIALS as inline JSON: {e}")
            return None

    if not os.path.exists(cred_path):
        print(f"[WARNING] Service account key file not found at: {cred_path}. Clearing environment variable and using default ADC.")
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        return None

    print(f"[INFO] Loading service account credentials from: {cred_path}")
    try:
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as e:
        print(f"[ERROR] Failed to load credentials from file: {e}")
        return None


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Initialize ChatGoogleGenerativeAI with Vertex AI configuration and credentials."""
    # Ensure Vertex AI mode is enabled to support Service Account authentication
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    
    if not project:
        # Fall back to a default project if not set (or prompt user if needed, but here we prioritize default)
        project = "restaurant-c1836"
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
        print(f"[WARNING] GOOGLE_CLOUD_PROJECT was not set. Defaulting to '{project}'")
        
    credentials = get_credentials()
    
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        project=project,
        location=location,
        credentials=credentials,
        temperature=temperature
    )


@tool
def search_inventory(query: str) -> str:
    """Search the pharmacy's inventory database for medications.
    
    Returns details about availability, price, stock level, prescription requirements,
    and any designated safe over-the-counter (OTC) alternatives.
    
    Args:
        query: The medication name or search term.
    """
    print(f"[TOOL] search_inventory called with query='{query}'")
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        return "Error: DATABASE_URL not set."
    
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT 
                        i1.id, i1.name, i1.dosage, i1.form, i1.price, i1.stock, i1.is_prescription,
                        i2.name AS alt_name, i2.dosage AS alt_dosage, i2.form AS alt_form, i2.price AS alt_price, i2.stock AS alt_stock
                    FROM inventory i1
                    LEFT JOIN inventory i2 ON i1.safe_otc_alternative_id = i2.id
                    WHERE i1.name ILIKE %s OR i1.name ILIKE %s;
                """, (f"%{query}%", f"{query}%"))
                rows = cur.fetchall()
                
                if not rows:
                    return f"No medications found matching '{query}' in our database."
                
                results = []
                for row in rows:
                    alt_info = "None"
                    if row["alt_name"]:
                        alt_info = f"{row['alt_name']} {row['alt_dosage']} {row['alt_form']} (Price: ${row['alt_price']:.2f}, Stock: {row['alt_stock']})"
                    
                    res = (
                        f"ID: {row['id']}\n"
                        f"Medication: {row['name']} {row['dosage']} {row['form']}\n"
                        f"Stock Level: {row['stock']}\n"
                        f"Price per unit: ${row['price']:.2f}\n"
                        f"Prescription Required: {'Yes' if row['is_prescription'] else 'No'}\n"
                        f"Safe OTC Alternative: {alt_info}"
                    )
                    results.append(res)
                return "\n".join(results)
    except Exception as e:
        return f"Error querying database: {e}"


@tool
def calculate_total(items: List[Dict[str, Any]]) -> str:
    """Calculate the subtotal, sales tax, handling fee, and grand total for a list of medication items.
    
    Args:
        items: A list of dicts containing medication details.
               Each dict must have:
               - 'name' (str): name of the medication
               - 'quantity' (int): quantity being ordered
               - 'price' (float): unit price of the medication
    """
    print(f"[TOOL] calculate_total called with items={items}")
    subtotal = 0.0
    breakdown = []
    for item in items:
        name = item.get("name", "Unknown Item")
        qty = int(item.get("quantity", 0))
        price = float(item.get("price", 0.0))
        item_total = qty * price
        subtotal += item_total
        breakdown.append(f"- {name}: {qty} x ${price:.2f} = ${item_total:.2f}")
        
    tax_rate = 0.08
    tax = subtotal * tax_rate
    fee = 1.50
    grand_total = subtotal + tax + fee
    
    result = (
        "Price Calculation Summary:\n" +
        "\n".join(breakdown) + "\n"
        f"Subtotal: ${subtotal:.2f}\n"
        f"Sales Tax (8%): ${tax:.2f}\n"
        f"Processing/Handling Fee: ${fee:.2f}\n"
        f"Grand Total Due: ${grand_total:.2f}"
    )
    return result


@tool
def process_payment(payment_method: str, total_amount: float) -> str:
    """Process the payment of the specified total amount via the selected payment method.
    
    This tool requires external human-in-the-loop payment confirmation.
    
    Args:
        payment_method: The payment method chosen by the customer (e.g., 'Credit Card', 'Mobile Money', 'Cash').
        total_amount: The total amount to be charged.
    """
    print(f"[TOOL] process_payment called with method='{payment_method}', amount={total_amount}")
    import random
    txn_id = f"TXN-{random.randint(100000, 999999)}"
    return (
        f"Payment Successful!\n"
        f"Transaction ID: {txn_id}\n"
        f"Method: {payment_method}\n"
        f"Amount Paid: ${total_amount:.2f}\n"
        f"Status: Confirmed"
    )


def create_langchain_agent(checkpointer=None):
    """Create the Pharmacy Assistant AI agent with Gemini model, tools, and human-in-the-loop configurations."""
    llm = get_llm()
    tools = [search_inventory, calculate_total, process_payment]
    
    system_prompt = (
        "You are a highly professional, accurate, and polite AI Pharmacy Assistant. "
        "Your primary role is to help customers efficiently process medication requests. "
        "You must be empathetic, precise, and entirely focused on customer service and inventory management.\n\n"
        
        "STRICT MEDICAL GUARDRAIL:\n"
        "You are not a doctor or a pharmacist. You must NEVER provide medical advice, diagnose symptoms, "
        "or suggest alternative prescription medications. If a user asks for medical advice or suggestions, "
        "kindly direct them to consult a licensed pharmacist or physician.\n\n"
        
        "YOUR STEP-BY-STEP WORKFLOW:\n"
        "1. Intake & Extraction: Read the medications, dosages, and quantities requested from the customer's text or "
        "prescribed in the uploaded image. If the handwriting/text is illegible or ambiguous, pause and ask the user for clarification.\n"
        "2. Database Search & Inventory Check: Query the inventory for each item using search_inventory. Verify exact match for medication name, dosage, and form.\n"
        "3. Availability & Pricing Communication: Clear list of items with Name & Dosage, Status (In Stock/Out of Stock), and Price (if in stock). "
        "If an item is out of stock, inform the customer. NEVER suggest alternatives for prescription drugs. For OTC products, "
        "you may list/suggest the exact generic equivalent ONLY IF the search_inventory database flags it as a safe OTC alternative. "
        "Ask the customer if they would like to proceed with the available items.\n"
        "4. Total Calculation: Once the user confirms they want to proceed, calculate the total using calculate_total. "
        "Provide a clear subtotal, tax, fee, and grand total. Do NOT calculate the math yourself; always use calculate_total.\n"
        "5. Payment Processing: Present the bill. Prompt the user for payment by providing accepted methods (Credit Card, Mobile Money, Cash) and "
        "the payment link: 'Your total comes to [Total Amount]. Would you like to pay via Credit Card, Mobile Money, or Cash at the counter? Please click the payment link below to proceed.' "
        "Then call process_payment. This tool will trigger a payment authorization pause.\n"
        "6. Final Confirmation: Once the transaction is confirmed, generate a digital receipt showing the details and provide pickup/delivery instructions."
    )
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "process_payment": {"allowed_decisions": ["approve", "reject"]},
                }
            )
        ]
    )


def _run_loop(agent, config):
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting agent session. Goodbye!")
                break
            
            # Check for image input simulation
            if user_input.lower().startswith("image:") or (
                os.path.exists(user_input) and user_input.lower().endswith((".png", ".jpg", ".jpeg"))
            ):
                image_path = user_input
                if image_path.lower().startswith("image:"):
                    image_path = image_path[6:].strip()
                
                if not os.path.exists(image_path):
                    print(f"\n[ERROR] Image file not found: {image_path}")
                    continue
                
                print(f"Loading and encoding image: {image_path}...")
                try:
                    ext = "png" if image_path.lower().endswith(".png") else "jpeg"
                    with open(image_path, "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    content = [
                        {"type": "text", "text": "Please analyze this prescription image and extract the medication name, dosage, and quantity."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64_data}"}}
                    ]
                except Exception as e:
                    print(f"\n[ERROR] Failed to read image: {e}")
                    continue
            else:
                content = user_input
                
            print("Agent is thinking...")
            result = agent.invoke(
                {"messages": [{"role": "user", "content": content}]},
                config=config
            )
            
            # Resolve any interrupts
            while "__interrupt__" in result:
                interrupt_val = result["__interrupt__"]
                print(f"\n[INTERRUPT] Payment approval required.")
                print(f"Details: {interrupt_val}")
                
                action = input("\n[Payment Gateway] Enter 'pay' to approve, 'cancel' to reject: ").strip().lower()
                if action == "pay":
                    from langgraph.types import Command
                    print("Processing payment confirmation...")
                    result = agent.invoke(
                        Command(resume={"decisions": [{"type": "approve"}]}),
                        config=config
                    )
                elif action == "cancel":
                    from langgraph.types import Command
                    print("Cancelling payment...")
                    result = agent.invoke(
                        Command(resume={"decisions": [{"type": "reject", "feedback": "Payment was cancelled/declined by the user."}]}),
                        config=config
                    )
                else:
                    print("Invalid input. Please enter 'pay' or 'cancel'.")
                    continue
            
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"\nAgent: {last_msg.content}")
            else:
                print("\nAgent: [No response message]")
                
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")


def run_cli():
    """Run an interactive CLI session with the agent, supporting persistent checkpointer."""
    print("=" * 60)
    print("      AI Pharmacy Assistant CLI")
    print("=" * 60)
    
    thread_id = input("Enter Thread ID (default: pharmacy-thread-1): ").strip()
    if not thread_id:
        thread_id = "pharmacy-thread-1"
        
    print(f"\nInitializing agent with Postgres persistence (thread_id: {thread_id})...")
    
    try:
        from checkpointer import get_sync_checkpointer, close_sync_pool
        has_checkpointer = True
    except ImportError:
        print("[WARNING] Could not load checkpointer.py. Running without Postgres saver.")
        agent = create_langchain_agent()
        has_checkpointer = False
        
    config = {"configurable": {"thread_id": thread_id}}
    
    print("\nAgent ready! Type 'exit' or 'quit' to end session.")
    print("You can upload a prescription image by entering its file path directly.")
    print("-" * 60)
    
    if has_checkpointer:
        with get_sync_checkpointer() as checkpointer:
            agent = create_langchain_agent(checkpointer=checkpointer)
            _run_loop(agent, config)
        close_sync_pool()
    else:
        _run_loop(agent, config)


if __name__ == "__main__":
    run_cli()
