import base64
import io
import json
import os

import requests
from PIL import Image, ImageDraw
from rich import print as rprint

NINE_ROUTER_BASE_URL = os.getenv("NINE_ROUTER_BASE_URL", "http://localhost:20128/v1").rstrip("/")
NINE_ROUTER_API_KEY = os.getenv("NINE_ROUTER_API_KEY", "")
TARGET_MODEL = os.getenv("NINE_ROUTER_MODEL", "qwen2.5-vl")
PROMPT = "Locate the cat and return only its bounding box as [x1, y1, x2, y2]. Coordinates must be integer pixel coordinates in the original image."


def load_pic(path: str) -> io.BytesIO:
    img = Image.open(path)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")

    return buffered


def get_pic_base64(image: io.BytesIO) -> str:
    img_bytes = image.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def get_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def normalize_bounding_box(
    bounding_box: list[int | float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounding_box
    max_x = max(x1, x2)
    max_y = max(y1, y2)

    if max_x <= 1 and max_y <= 1:
        x1, x2 = x1 * image_width, x2 * image_width
        y1, y2 = y1 * image_height, y2 * image_height
    elif max_x <= 1000 and max_y <= 1000 and (
        max_x > image_width
        or max_y > image_height
        or image_width > 1000
        or image_height > 1000
    ):
        x1, x2 = x1 * image_width / 1000, x2 * image_width / 1000
        y1, y2 = y1 * image_height / 1000, y2 * image_height / 1000

    x1, x2 = sorted((round(x1), round(x2)))
    y1, y2 = sorted((round(y1), round(y2)))
    return (
        max(0, min(x1, image_width - 1)),
        max(0, min(y1, image_height - 1)),
        max(0, min(x2, image_width - 1)),
        max(0, min(y2, image_height - 1)),
    )


def draw_bounding_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image: io.BytesIO,
) -> str:
    to_draw = Image.open(image)
    draw = ImageDraw.Draw(to_draw)

    draw.rectangle([x1, y1, x2, y2], outline="green", width=9)

    to_draw.save("/home/torham/Pictures/detected_cat.png")

    return "/home/torham/Pictures/detected_cat.png"


def main(pic_path: str, prompt: str = PROMPT) -> dict[str, str | float | int]:

    image_width, image_height = get_image_size(pic_path)
    prompt = (
        f"{prompt} The original image size is {image_width}x{image_height} pixels. "
        "Do not use normalized coordinates or a 0-1000 coordinate system."
    )
    body = {
        "model": TARGET_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{get_pic_base64(load_pic(pic_path))}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "reasoning": {"effort": "none"},
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
    rprint(data)
    usage = data.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content")
    if not content:
        finish_reason = choice.get("finish_reason", "unknown")
        reasoning = message.get("reasoning", "")
        raise RuntimeError(
            "The model did not return a bounding box. "
            f"finish_reason={finish_reason!r}, reasoning={reasoning!r}. "
            "Try a vision model with reasoning disabled or increase max_tokens."
        )

    return {
        "response": content,
        "model": data.get("model", TARGET_MODEL),
        "total_duration": 0.0,
        "prompt_eval_duration": 0.0,
        "eval_duration": 0.0,
        "eval_count": completion_tokens,
        "token_per_second": 0.0,
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

    try:
        bounding_box = json.loads(result["response"])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"The model returned invalid bounding-box JSON: {result['response']!r}"
        ) from error

    if (
        not isinstance(bounding_box, list)
        or len(bounding_box) != 4
        or not all(isinstance(value, (int, float)) for value in bounding_box)
    ):
        raise RuntimeError(
            f"The model returned an invalid bounding box: {bounding_box!r}. "
            "Expected [x1, y1, x2, y2]."
        )

    bounding_box = normalize_bounding_box(
        bounding_box, image_width=image_width, image_height=image_height
    )
    boxed_pic = draw_bounding_box(
        x1=bounding_box[0],
        y1=bounding_box[1],
        x2=bounding_box[2],
        y2=bounding_box[3],
        image=load_pic(pic_address),
    )
    print(response)
    print(f"bounding boxed picture saved at: {boxed_pic}")
