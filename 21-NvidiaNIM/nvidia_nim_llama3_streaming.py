import os
from dotenv import load_dotenv
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

completion = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[{"role": "user", "content": "hello"}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True,
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()
