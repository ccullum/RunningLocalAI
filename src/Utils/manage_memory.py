import os
import sys
from qdrant_client import QdrantClient

# ==========================================
# PATH INJECTION & SETUP
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.colors import Colors

# Dynamically find the database
data_dir = os.path.abspath(os.path.join(src_dir, "..", "data"))
qdrant_path = os.path.join(data_dir, "qdrant_storage")
collection = "jarvis_memory"

print(f"{Colors.SYSTEM}Connecting to Persistent Memory at: {qdrant_path}{Colors.RESET}\n")

try:
    client = QdrantClient(path=qdrant_path)
    if not client.collection_exists(collection):
        print(f"{Colors.ERROR}Memory collection does not exist yet.{Colors.RESET}")
        sys.exit()
except Exception as e:
    print(f"{Colors.ERROR}Failed to connect: {e}{Colors.RESET}")
    sys.exit()

def display_memories():
    """Scrolls through the database and prints all saved memories with their IDs."""
    print(f"{Colors.JARVIS}--- CURRENT LONG-TERM MEMORIES ---{Colors.RESET}")
    # We use scroll() to fetch records without needing a vector search
    records, next_page_offset = client.scroll(
        collection_name=collection,
        limit=100, # Adjust if you get more than 100 memories
        with_payload=True,
        with_vectors=False
    )
    
    if not records:
        print(f"{Colors.WARNING}The database is completely empty.{Colors.RESET}")
        return False

    for record in records:
        text = record.payload.get('text', 'No text found')
        print(f"{Colors.MEMORY}[ID: {record.id}]{Colors.RESET}")
        print(f"   Text: {text}\n")
    return True

def main():
    while True:
        has_records = display_memories()
        if not has_records:
            break
            
        print(f"{Colors.SYSTEM}Enter the exact ID of the memory you want to delete (or type 'exit' to quit):{Colors.RESET}")
        user_choice = input("> ").strip()
        
        if user_choice.lower() in ['exit', 'quit', 'q']:
            print("Exiting Memory Manager.")
            client.close()
            break
            
        try:
            # Delete the specific point from Qdrant
            client.delete(
                collection_name=collection,
                points_selector=[user_choice]
            )
            print(f"{Colors.USER}Successfully deleted memory: {user_choice}{Colors.RESET}\n")
        except Exception as e:
            print(f"{Colors.ERROR}Failed to delete (check the ID format): {e}{Colors.RESET}\n")

if __name__ == "__main__":
    main()