from google import genai
from dotenv import load_dotenv
from pathlib import Path
import os

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# Environment setup
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# Create client
client = genai.Client(api_key=API_KEY)

# List files
print("\n========== Listing Files ==========")

stored_files = list(client.files.list(config={'page_size': 50}))

if not stored_files:
    print("No files found.")
else:
    for f in stored_files:
        print(f"  Name:         {f.name}")
        print(f"  Display Name: {f.display_name}")
        print(f"  Size:         {f.size_bytes} bytes")
        print(f"  URI:          {f.uri}")
        print(f"  State:        {f.state.name}")
        print("-" * 20)

# Delete a file
print("\n========== Deleting All Files ==========")
for f in stored_files:
    client.files.delete(name=f.name)
    print(f"Deleted {f.name}")
print("All listed files have been deleted.")

print("\nFile management script finished.")