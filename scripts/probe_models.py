import os, json, base64, io, time
from openai import OpenAI
from PIL import Image
from hastabench.bench import make_client
c = make_client(timeout=120)
im = Image.open("data/targets_v1/pataka_front.png").resize((256, 256)); b = io.BytesIO(); im.save(b, "PNG")
url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
tools = [{"type": "function", "function": {"name": "answer", "description": "report", "parameters": {"type": "object", "properties": {"fingers_extended": {"type": "integer"}}, "required": ["fingers_extended"]}}}]
for m in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "aws/claude-opus-4-8", "azure/gpt-5.6-sol", "azure/gpt-5.6-luna-ncf", "Azure/gpt-5.2-2025-12-11", "azure/gpt-5-mini", "gcp/gemini-3.8-flash", "gcp/gemini-2.5-pro", "gcp/gemini-3.1-flash-lite", "azure/Kimi-K2.5", "coreweave/glmv5", "coreweave/glmv52", "aws/glm-5"]:
    t = time.time()
    try:
        r = c.chat.completions.create(model=m, messages=[{"role": "user", "content": [{"type": "text", "text": "How many fingers (excluding thumb) are extended in this hand? Use the tool."}, {"type": "image_url", "image_url": {"url": url}}]}], tools=tools, tool_choice="auto")
        msg = r.choices[0].message
        print(json.dumps(dict(model=m, ok=True, tool=[x.function.arguments for x in (msg.tool_calls or [])], text=(msg.content or "")[:80], usage=[r.usage.prompt_tokens, r.usage.completion_tokens], s=round(time.time() - t, 1))), flush=True)
    except Exception as e:
        print(json.dumps(dict(model=m, ok=False, err=str(e)[:200])), flush=True)
