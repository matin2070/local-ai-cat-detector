import requests

VLM_ADDRESS = "http://localhost:11434/"
API_ROUTE = "api/generate"
TARGET_MODEL = "llama3.2:3b"


def get_prompt() -> str:
    return input("Enter your prompt: ")


def main(prompt: str) -> dict[str, str | float | int]:

    body = {"model": TARGET_MODEL, "stream": False, "prompt": prompt}

    print(f"Requesting to {TARGET_MODEL}...")
    result = requests.post(url=f"{VLM_ADDRESS}{API_ROUTE}", json=body)
    if result.status_code != 200:
        raise ValueError(f"Wrong status code! {result.status_code}")

    data = result.json()
    return {
        "response": data["response"],
        "model": data["model"],
        "total_duration": data["total_duration"] / 1e9,
        "prompt_eval_duration": data["prompt_eval_duration"] / 1e9,
        "eval_duration": data["eval_duration"] / 1e9,
        "eval_count": int(data["eval_count"]),
        "token_per_second": data["eval_count"] / (data["eval_duration"] / 1e9),
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
