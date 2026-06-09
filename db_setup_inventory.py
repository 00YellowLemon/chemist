import os
import sys
import psycopg
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def setup_inventory():
    if not DATABASE_URL:
        print("Error: DATABASE_URL not set in .env file", file=sys.stderr)
        sys.exit(1)

    print("Connecting to the database to set up inventory table...")
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Drop existing table if needed (to ensure clean seeding)
                print("Creating inventory table...")
                cur.execute("DROP TABLE IF EXISTS inventory CASCADE;")
                
                cur.execute("""
                    CREATE TABLE inventory (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        dosage VARCHAR(50) NOT NULL,
                        form VARCHAR(50) NOT NULL,
                        price NUMERIC(10, 2) NOT NULL,
                        stock INTEGER NOT NULL,
                        is_prescription BOOLEAN DEFAULT FALSE,
                        safe_otc_alternative_id INTEGER REFERENCES inventory(id) ON DELETE SET NULL
                    );
                """)
                
                # We will insert the base items first, then update generic alternatives if any.
                print("Seeding medication items...")
                
                items = [
                    # name, dosage, form, price, stock, is_prescription
                    ("Amoxicillin", "500mg", "capsule", 0.50, 20, True),
                    ("Ibuprofen", "400mg", "tablet", 0.45, 10, False),
                    ("Acetaminophen", "500mg", "tablet", 0.15, 100, False),
                    ("Loratadine", "10mg", "tablet", 0.80, 0, False), # out of stock
                    ("Cetirizine", "10mg", "tablet", 0.90, 50, False),
                    ("Metformin", "500mg", "tablet", 0.30, 0, True),  # out of stock
                ]
                
                inserted_ids = {}
                for name, dosage, form, price, stock, is_prescription in items:
                    cur.execute("""
                        INSERT INTO inventory (name, dosage, form, price, stock, is_prescription)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (name, dosage, form, price, stock, is_prescription))
                    item_id = cur.fetchone()[0]
                    inserted_ids[f"{name} {dosage}"] = item_id

                # Link Loratadine 10mg to Cetirizine 10mg as a safe OTC alternative
                loratadine_id = inserted_ids.get("Loratadine 10mg")
                cetirizine_id = inserted_ids.get("Cetirizine 10mg")
                
                if loratadine_id and cetirizine_id:
                    print(f"Linking Loratadine 10mg (id={loratadine_id}) to safe OTC alternative Cetirizine 10mg (id={cetirizine_id})")
                    cur.execute("""
                        UPDATE inventory 
                        SET safe_otc_alternative_id = %s 
                        WHERE id = %s;
                    """, (cetirizine_id, loratadine_id))

                conn.commit()
                print("Database setup and seeding completed successfully!")
                
    except Exception as e:
        print(f"Failed to setup inventory table: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    setup_inventory()
