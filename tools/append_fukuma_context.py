import json
import os
import shutil

PROMPT_JSON = os.path.join("templates", "son", "prompt.json")
CONTEXT_FILE = os.path.join("templates", "son", "fukuma_sun_latex.txt")

if not os.path.exists(PROMPT_JSON):
    raise FileNotFoundError(f"prompt.json not found: {PROMPT_JSON}")
if not os.path.exists(CONTEXT_FILE):
    raise FileNotFoundError(f"context file not found: {CONTEXT_FILE}")

with open(PROMPT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
    content = f.read()

marker = f"BEGIN CONTEXT ({CONTEXT_FILE})"
end_marker = "END CONTEXT"

td = data.get("task_description", "")
if marker not in td:
    shutil.copyfile(PROMPT_JSON, PROMPT_JSON + ".bak")
    block = "\n\n" + marker + "\n" + content + "\n" + end_marker + "\n"
    data["task_description"] = td + block
    with open(PROMPT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Appended context to {PROMPT_JSON}. Backup at {PROMPT_JSON}.bak")
else:
    print("Context already present; no changes made.")

