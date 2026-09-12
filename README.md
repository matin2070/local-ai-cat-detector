# 🐈 Local AI Cat Detector

> **Where is the cat?**

A small experimental project for learning how to run **AI vision models locally** with Ollama and Python.

The goal is simple:

**Give a local Vision-Language Model (VLM) an image → ask it where the cat is → get a bounding box → draw the box on the image.**

No cloud API is required for inference.

---

## 🎥 What this project is about

This project was built as a hands-on way to understand the building blocks behind local AI applications.

Along the way, it covers:

* What **LLMs** and **VLMs** are
* What **Ollama** does
* Running AI models locally
* Talking to Ollama through its REST API
* Sending images to a vision model
* Base64 image encoding
* Asking a VLM for a bounding box
* Drawing the result with Pillow
* Measuring model inference performance
* Understanding model loading, CPU/GPU usage, and warm vs. cold inference

The project is intentionally small so that the individual pieces are easy to understand.

---

## 🧠 LLM vs VLM

### LLM — Large Language Model

An LLM primarily works with text.

For example:

```text
User: What is a cat?

Model: A cat is a small domesticated mammal...
```

This project initially uses `llama3.2:3b` to demonstrate communication with a local language model.

### VLM — Vision-Language Model

A VLM can work with both **text and images**.

For example:

```text
Image + "Where is the cat?"
             ↓
            VLM
             ↓
      [x1, y1, x2, y2]
```

The project uses:

**Qwen2.5-VL 3B**

The model looks at the image and returns the location of the cat as a bounding box.

---

## 🦙 What is Ollama?

Ollama makes it easy to run AI models locally.

Instead of sending an image to a remote AI API, the application communicates with a local Ollama server:

```text
Python application
       │
       │ HTTP
       ▼
┌─────────────────┐
│     Ollama      │
│ localhost:11434 │
└────────┬────────┘
         │
         ▼
   Qwen2.5-VL 3B
         │
         ▼
    Model output
```

This means the inference can happen entirely on your own machine.

---

## 🚀 Requirements

You'll need:

* Linux, macOS, or Windows
* Python 3.11+
* Ollama
* Enough RAM/VRAM to run the selected model
* An image containing a cat

The project was developed and tested on relatively old hardware:

* **CPU:** Intel Core i7-6700HQ
* **RAM:** 16 GB
* **GPU:** NVIDIA GTX 960M
* **VRAM:** 4 GB

The VLM runs using a combination of CPU and GPU resources.

---

## 📦 Installation

### 1. Install Ollama

Install Ollama from:

https://ollama.com/

Verify that it is available:

```bash
ollama --version
```

---

### 2. Download the models

For the language-model experiments:

```bash
ollama pull llama3.2:3b
```

For image understanding:

```bash
ollama pull qwen2.5vl:3b
```

You can verify the installed models with:

```bash
ollama list
```

---

## 🐍 Python setup

This project uses `uv` for Python project management.

Clone the repository:

```bash
git clone https://github.com/TorhamDev/local-ai-cat-detector.git
cd local-ai-cat-detector
```

Install the dependencies:

```bash
uv sync
```

Then run the application:

```bash
uv run python main.py
```

The program will ask for an image path.

---

## 🔍 How the detector works

The core pipeline is:

```text
             Image
               │
               ▼
        Python / Pillow
               │
               │ base64
               ▼
        ┌──────────────┐
        │    Ollama    │
        └──────┬───────┘
               │
               ▼
        Qwen2.5-VL 3B
               │
               │ bounding box
               ▼
       [x1, y1, x2, y2]
               │
               ▼
        Python receives
        the coordinates
               │
               ▼
          Pillow draws
         bounding box
```

The VLM doesn't draw the rectangle.

It predicts four numbers representing the bounding box.

Python then takes those numbers and passes them to Pillow to draw the rectangle.

---

## 📐 Bounding boxes

The model returns four coordinates:

```text
[x1, y1, x2, y2]
```

These represent the two corners of the bounding box:

```text
(x1, y1)
    ┌────────────────┐
    │                │
    │      CAT       │
    │                │
    └────────────────┘
                 (x2, y2)
```

For example, the model might return:

```text
[230, 110, 630, 780]
```

The coordinates are then passed to Pillow, which draws the rectangle around the detected cat.

> **Note:** The current implementation intentionally keeps the bounding-box handling simple. There is currently no additional coordinate validation or normalization layer. This may be improved in a future version.

---

## ⚡ Performance

One of the interesting experiments in this project was measuring inference time.

The first request can be significantly slower because the model needs to be loaded into memory.

Once the model is already loaded, subsequent requests are much faster.

Conceptually:

```text
Cold request

Request
  │
  ├── Load model       ← expensive
  ├── Process image
  └── Generate output


Warm request

Request
  │
  ├── Model already loaded
  ├── Process image
  └── Generate output  ← much faster
```

Ollama provides timing information such as:

* model load duration
* prompt evaluation duration
* generation duration
* generated token count

This makes it possible to see what the model is actually spending time doing.

---

## 🖥️ Local AI

One of the main reasons for building this project was to understand what is actually happening when an AI model runs locally.

There is no magic API call like:

```text
image → mysterious cloud → result
```

Instead, the application is essentially:

```text
Python
  ↓
HTTP request
  ↓
Ollama
  ↓
Local model
  ↓
Inference
  ↓
HTTP response
  ↓
Python
```

That makes the entire process much easier to inspect and experiment with.

---

## 🛠️ Project status

This is primarily a **learning project** rather than a production-ready object detector.

The VLM is being used for experimentation and understanding local multimodal AI.

The current implementation focuses on detecting a single cat and drawing its bounding box.

Possible future experiments include:

* Bounding-box validation
* Coordinate normalization
* Detecting multiple cats
* Returning multiple bounding boxes
* Structured JSON output
* Video input
* Real-time webcam detection
* Performance benchmarking
* Comparing different VLMs
* Running the model with different quantizations
* Exploring GPU/CPU offloading
* Building a small web interface

---

## 📚 Useful resources

* [Ollama](https://ollama.com/)
* [Ollama Vision documentation](https://docs.ollama.com/capabilities/vision)
* [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)
* [Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/)

---

## ⭐ Why build this?

Because the best way to understand local AI is to actually build something with it.

Start with:

> **"Where is the cat?"**

And end up learning about:

> **LLMs → VLMs → Ollama → APIs → image encoding → inference → tokens → GPU memory → bounding boxes → image processing**

That's the fun part.

---

## 📄 License

This project is provided for learning and experimentation.
