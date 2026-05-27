import os
from openai import OpenAI

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

MODELS = [
    ("1", "gpt-4o",                  "OpenAI GPT-4o (most capable)"),
    ("2", "gpt-4o-mini",             "OpenAI GPT-4o Mini (fast & cheap)"),
    ("3", "Meta-Llama-3.1-70B-Instruct",  "Meta Llama 3.1 70B"),
    ("4", "Meta-Llama-3.1-8B-Instruct",   "Meta Llama 3.1 8B (lightweight)"),
    ("5", "Mistral-large",           "Mistral Large"),
    ("6", "Mistral-small",           "Mistral Small"),
    ("7", "Phi-3.5-mini-instruct",   "Microsoft Phi-3.5 Mini"),
    ("8", "Cohere-command-r-plus",   "Cohere Command R+"),
]

def pick_model():
    print("\nAvailable models:")
    print("-" * 50)
    for num, model_id, description in MODELS:
        print(f"  {num}. {description}")
        print(f"     ({model_id})")
    print("-" * 50)

    while True:
        choice = input("Pick a model (1-8) or press Enter for gpt-4o-mini: ").strip()
        if choice == "":
            return "gpt-4o-mini"
        for num, model_id, _ in MODELS:
            if choice == num:
                return model_id
        print("Invalid choice, please enter a number between 1 and 8.")

def chat():
    print("=" * 50)
    print("        AI Chatbot — GitHub Models")
    print("=" * 50)

    model = pick_model()
    print(f"\nUsing model: {model}")
    print("Type 'quit' to exit, 'model' to switch models.\n")

    messages = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "model":
            model = pick_model()
            print(f"Switched to: {model}\n")
            continue

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"AI: {reply}\n")

if __name__ == "__main__":
    chat()
