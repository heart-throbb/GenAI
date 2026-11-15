import requests
import gradio as gr

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "codeguru"


def generate_response(message, history):
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    payload = {"model": MODEL_NAME, "messages": messages, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Make sure it is running (`ollama serve`)."
    except requests.exceptions.Timeout:
        return "The request timed out. The model may still be loading, so try again."
    except requests.exceptions.HTTPError:
        return f"Ollama returned an error: {response.text}"


demo = gr.ChatInterface(
    fn=generate_response,
    type="messages",
    title="CodeGuru",
    description="Chat with your local Ollama model.",
)

if __name__ == "__main__":
    demo.launch()
