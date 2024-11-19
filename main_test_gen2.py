import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

# Step 1: Create Example Dataset
def create_dataset():
    data = {
        "prompt": [
            "function that returns the square of a number",
            "function to sum two numbers",
            "create a class with a method to reverse a string",
            "function that checks if a number is prime",
            "function to compute the Fibonacci sequence",
            "function to find the maximum of three numbers",
            "function to concatenate two strings",
            "function to find all duplicates in a list",
            "function to check if a string is a palindrome",
            "function to find the factorial of a number",
            "function to sort a list of numbers",
            "function to calculate the length of a list",
            "function to convert a list of strings to uppercase",
            "function to calculate the average of a list of numbers",
            "create a class for complex numbers with addition",
            "function to flatten a nested list",
            "function to get unique elements from a list",
            "function to perform bubble sort",
            "function to implement binary search",
            "function to calculate GCD of two numbers",
            "function to transpose a matrix",
            "function to calculate n-th root of a number",
            "function to check if a year is a leap year",
            "function to find the longest word in a string",
            "function to compute sum of digits in a number",
            "function to replace vowels in a string",
            "function to simulate rolling two dice",
            "function to calculate the area of a circle",
            "function to generate a random password",
            "create a function to convert Celsius to Fahrenheit",
            "function to remove punctuation from a string",
            "function to compute the power of a number"
        ],
        "code": [
            "def square(x):\n    return x * x",
            "def sum(a, b):\n    return a + b",
            "class StringManipulator:\n    def reverse(self, s):\n        return s[::-1]",
            "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            "def fibonacci(n):\n    sequence = [0, 1]\n    for i in range(2, n):\n        sequence.append(sequence[-1] + sequence[-2])\n    return sequence[:n]",
            "def max_of_three(a, b, c):\n    return max(a, b, c)",
            "def concat_strings(s1, s2):\n    return s1 + s2",
            "def find_duplicates(lst):\n    return list(set([x for x in lst if lst.count(x) > 1]))",
            "def is_palindrome(s):\n    return s == s[::-1]",
            "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
            "def sort_numbers(lst):\n    return sorted(lst)",
            "def list_length(lst):\n    return len(lst)",
            "def to_uppercase(lst):\n    return [s.upper() for s in lst]",
            "def average(lst):\n    return sum(lst) / len(lst)",
            "class ComplexNumber:\n    def __init__(self, real, imag):\n        self.real = real\n        self.imag = imag\n    def __add__(self, other):\n        return ComplexNumber(self.real + other.real, self.imag + other.imag)",
            "def flatten(lst):\n    return [item for sublist in lst for item in sublist]",
            "def unique(lst):\n    return list(set(lst))",
            "def bubble_sort(lst):\n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if lst[j] > lst[j+1]:\n                lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst",
            "def binary_search(lst, target):\n    low, high = 0, len(lst) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if lst[mid] < target:\n            low = mid + 1\n        elif lst[mid] > target:\n            high = mid - 1\n        else:\n            return mid\n    return -1",
            "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
            "def transpose(matrix):\n    return list(map(list, zip(*matrix)))",
            "def nth_root(x, n):\n    return x ** (1 / n)",
            "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)",
            "def longest_word(s):\n    words = s.split()\n    return max(words, key=len)",
            "def sum_of_digits(n):\n    return sum(int(digit) for digit in str(n))",
            "def replace_vowels(s, replacement):\n    vowels = 'aeiouAEIOU'\n    return ''.join([replacement if c in vowels else c for c in s])",
            "def roll_dice():\n    import random\n    return random.randint(1, 6), random.randint(1, 6)",
            "def area_of_circle(radius):\n    from math import pi\n    return pi * radius ** 2",
            "def generate_password(length=8):\n    import random\n    import string\n    characters = string.ascii_letters + string.digits + string.punctuation\n    return ''.join(random.choice(characters) for i in range(length))",
            "def celsius_to_fahrenheit(celsius):\n    return (celsius * 9/5) + 32",
            "def remove_punctuation(s):\n    import string\n    return s.translate(str.maketrans('', '', string.punctuation))",
            "def power(base, exp):\n    return base ** exp"
        ]
    }
    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, val_df

# Step 2: Prepare Data Loader
def encode_data(examples, tokenizer, max_length=128):
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=max_length)
    labels = tokenizer(examples['code'], padding="max_length", truncation=True, max_length=max_length)
    inputs['labels'] = labels['input_ids']
    return inputs

def create_dataloader(dataset, tokenizer, batch_size=2):
    encoded_dataset = dataset.map(lambda x: encode_data(x, tokenizer), remove_columns=["prompt", "code"])
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True)

# Step 3: Training Loop
def train(train_loader, model, optimizer, device):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

def evaluate(val_loader, model, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    print(f"Validation Loss: {avg_loss}")

# Step 4: Generate Code from New Prompts
def generate_code(prompt, tokenizer, model, device, max_length=128):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        do_sample=True
    )
    generated_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_code

if __name__ == "__main__":
    train_df, val_df = create_dataset()

    # Load tokenizer and model
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create data loaders
    train_loader = create_dataloader(Dataset.from_pandas(train_df), tokenizer)
    val_loader = create_dataloader(Dataset.from_pandas(val_df), tokenizer)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Train the model
    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(train_loader, model, optimizer, device)
        evaluate(val_loader, model, device)

    # Test the model with a new prompt
    new_prompt = "function to calculate factorial of a number"
    generated_code = generate_code(new_prompt, tokenizer, model, device)
    print(f"Prompt: {new_prompt}")
    print(f"Generated Code: \n{generated_code}")
