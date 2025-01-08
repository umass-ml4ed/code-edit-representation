import os
from openai import OpenAI

# Ensure the API key is set in your environment or explicitly provide it
api_key = os.getenv("OPENAI_API_KEY")  # Fetch from environment variable
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

# Create the chat completion
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming in C."}
    ]
)

# Print the result
print(completion.choices[0].message.content)