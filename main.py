import base64
import io
import json

import requests
from PIL import Image, ImageDraw
from rich import print as rprint

OLLAMA_ADDRESS = "http://localhost:11434"
API_ROUTE = "/api/chat"
TARGET_MODEL = "qwen2.5vl:3b"
PROMPT = "Locate The Cat. Return only its bounding box as a list of [x1, y1, x2, y2]"


def load_pic(path: str) -> io.BytesIO:
    img = Image.open(path)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")

    return buffered


def get_pic_base64(image: io.BytesIO) -> str:
    img_bytes = image.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def draw_bounding_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image: io.BytesIO,
) -> str:
    to_draw = Image.open(image)
    draw = ImageDraw.Draw(to_draw)

    draw.rectangle([x1, y1, x2, y2], outline="green", width=29)

    to_draw.save("/home/torham/Pictures/detected_cat.png")

    return "/home/torham/Pictures/detected_cat.png"


def main(pic_path: str, prompt: str = PROMPT) -> dict[str, str | float | int]:

    body = {
        "model": TARGET_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (prompt),
                "images": [get_pic_base64(load_pic(pic_path))],
            }
        ],
        "options": {
            "num_predict": 30,
            "temperature": 0.0,
        },
        "stream": False,
        "keep_alive": -1,
    }

    print(f"Requesting to {TARGET_MODEL}...")
    result = requests.post(url=f"{OLLAMA_ADDRESS}{API_ROUTE}", json=body)
    result.raise_for_status()

    data = result.json()
    rprint(data)

    return {
        "response": data["message"]["content"],
        "model": data["model"],
        "total_duration": data["total_duration"] / 1e9,
        "prompt_eval_duration": data["prompt_eval_duration"] / 1e9,
        "eval_duration": data["eval_duration"] / 1e9,
        "eval_count": int(data["eval_count"]),
        "token_per_second": data["eval_count"] / (data["eval_duration"] / 1e9),
    }


if __name__ == "__main__":
    pic_address = input("Enter picture address: ")

    result = main(prompt=PROMPT, pic_path=pic_address)
    response = f"""
    Model: {result["model"]}
    Response: {result["response"]}
    Total time:{result["total_duration"]:.2f}s
    Generation time: {result["eval_duration"]:.2f}s
    Generated tokens: {result["eval_count"]}
    Speed: {result["token_per_second"]:.2f} tokens/s
    """

    bounding_box = json.loads(result["response"])
    boxed_pic = draw_bounding_box(
        x1=bounding_box[0],
        y1=bounding_box[1],
        x2=bounding_box[2],
        y2=bounding_box[3],
        image=load_pic(pic_address),
    )
    print(response)
    print(f"bounding boxed picture saved at: {boxed_pic}")
