import os
import requests

NINE_ROUTER_BASE_URL = os.getenv("NINE_ROUTER_BASE_URL", "http://localhost:20128/v1").rstrip("/")
NINE_ROUTER_API_KEY = os.getenv("NINE_ROUTER_API_KEY", "")
TARGET_MODEL = os.getenv("NINE_ROUTER_MODEL", "llama-3.2-3b")


def get_prompt() -> str:
    return input("Enter your prompt: ")


def main(prompt: str) -> dict[str, str | float | int]:

    body = {
        "model": TARGET_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
    }

    print(f"Requesting to {TARGET_MODEL}...")
    headers = {"Content-Type": "application/json"}
    if NINE_ROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {NINE_ROUTER_API_KEY}"

    result = requests.post(
        url=f"{NINE_ROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=body,
        timeout=120,
    )
    result.raise_for_status()
    data = result.json()
    usage = data.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))

    return {
        "response": data["choices"][0]["message"]["content"],
        "model": data.get("model", TARGET_MODEL),
        "total_duration": 0.0,
        "prompt_eval_duration": 0.0,
        "eval_duration": 0.0,
        "eval_count": completion_tokens,
        "token_per_second": 0.0,
    }


if __name__ == "__main__":
    result = main(prompt=get_prompt())
    response = f"""
    Model: {result["model"]}
    Response: {result["response"]}
    Total time:{result["total_duration"]:.2f}s
    Generation time: {result["eval_duration"]:.2f}s
    Generated tokens: {result["eval_count"]}
    Speed: {result["token_per_second"]:.2f} tokens/s
    """
    print(response)
