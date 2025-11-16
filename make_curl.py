import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="CSV file")
parser.add_argument("--text-col", required=True, help="Column with texts")
parser.add_argument("--capabilities", nargs="+", default=["classification"])
parser.add_argument("--config-path", required=True)
parser.add_argument("--url", default="http://localhost:9000/classify")
parser.add_argument("--output", default="output.json", help="Where curl will save API result")
parser.add_argument(
    "--save-to", default="generated_curl.sh", help="File to write the curl command"
)
args = parser.parse_args()

# Load CSV
df = pd.read_csv(args.input)

# Extract non-null texts
texts = df[args.text_col].dropna().astype(str).tolist()

# Build JSON body
payload = {
    "texts": texts,
    "capabilities": args.capabilities,
    "config_path": args.config_path,
}

# Pretty-print JSON
payload_str = json.dumps(payload, indent=2)

# Escape single quotes so JSON fits inside -d '...'
payload_str = payload_str.replace("'", "'\"'\"'")

# Build curl command
curl_cmd = (
    f"curl -X POST {args.url} \\\n"
    f'  -H "Content-Type: application/json" \\\n'
    f"  -d '{payload_str}' | jq '.' > {args.output}\n"
)

# Save curl command to file
Path(args.save_to).write_text(curl_cmd)

print(f"✔ Curl command written to: {args.save_to}")
