import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from flask import Flask, jsonify, request


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.4"))
REQUEST_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT_SEC", "30"))
ALLOWED_HOSTS = {"localhost", "127.0.0.1"}

app = Flask(__name__)


def _load_dotenv_file() -> None:
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv_file()


def _is_local_dev_origin(origin: str) -> bool:
    try:
        u = urlparse(origin or "")
    except Exception:
        return False
    return u.scheme in {"http", "https"} and (u.hostname in ALLOWED_HOSTS)


@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin", "")
    if _is_local_dev_origin(origin):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return ("", 204)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return jsonify({"ok": False, "error": "Missing OPENROUTER_API_KEY on server"}), 500

    payload = request.get_json(silent=True) or {}
    messages = payload.get("messages")
    model = payload.get("model") or DEFAULT_MODEL
    temperature = payload.get("temperature", DEFAULT_TEMPERATURE)

    if not isinstance(messages, list) or not messages:
        return jsonify({"ok": False, "error": "Field 'messages' must be a non-empty array"}), 400

    upstream_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    req = Request(
        OPENROUTER_URL,
        data=json.dumps(upstream_payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=REQUEST_TIMEOUT) as res:
            data = json.loads(res.read().decode("utf-8"))
    except HTTPError as e:
        return jsonify({"ok": False, "error": f"OpenRouter error ({e.code})"}), 502
    except URLError:
        return jsonify({"ok": False, "error": "Unable to reach OpenRouter"}), 502
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected server error"}), 500

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return jsonify({"ok": False, "error": "OpenRouter returned no choices"}), 502

    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content or "")

    if not text.strip():
        return jsonify({"ok": False, "error": "OpenRouter returned an empty message"}), 502

    return jsonify({"ok": True, "text": text})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
