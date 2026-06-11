from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model     = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
print("GPT-2 loaded successfully!\n")

# ──────────────────────────────────────────
# STEP 1: Text Generation Function
# ──────────────────────────────────────────
def generate_text(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ──────────────────────────────────────────
# STEP 2: Test Multiple Prompts
# ──────────────────────────────────────────
prompts = [
    "Artificial intelligence is changing the world by",
    "The future of medicine will involve",
    "Once upon a time in a land far away",
    "The most important skill in data science is",
    "Climate change is a global challenge because"
]

print("=" * 60)
print("GPT-2 TEXT GENERATION RESULTS")
print("=" * 60)

results = []
for i, prompt in enumerate(prompts):
    print(f"\nPrompt {i+1}: {prompt}")
    print("-" * 40)
    generated = generate_text(prompt)
    continuation = generated[len(prompt):]
    print(f"Generated: {continuation}")
    results.append({
        'prompt': prompt,
        'generated': generated,
        'continuation': continuation,
        'length': len(generated.split())
    })

# ──────────────────────────────────────────
# STEP 3: Analyze Token Probabilities
# ──────────────────────────────────────────
def get_token_probs(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        logits = outputs.logits
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)
    tokens = [tokenizer.decode([idx]) for idx in top_indices]
    return tokens, top_probs.numpy()

# ──────────────────────────────────────────
# STEP 4: Visualize Top Next Word Predictions
# ──────────────────────────────────────────
test_prompt = "Artificial intelligence is changing the world by"
tokens, probs = get_token_probs(test_prompt)

plt.figure(figsize=(10, 5))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tokens)))
bars = plt.barh(range(len(tokens)), probs, color=colors)
plt.yticks(range(len(tokens)), [f'"{t.strip()}"' for t in tokens])
plt.xlabel('Probability')
plt.title(f'Top 10 Next Word Predictions\nPrompt: "{test_prompt}"',
          fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
for bar, prob in zip(bars, probs):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{prob:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('token_probabilities.png')
plt.show()
print("\nSaved: token_probabilities.png")

# ──────────────────────────────────────────
# STEP 5: Temperature Comparison
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("TEMPERATURE ANALYSIS")
print("=" * 60)
print("(Lower temp = more focused, Higher temp = more creative)\n")

test_prompt = "The future of artificial intelligence"
temperatures = [0.3, 0.7, 1.0, 1.5]

temp_results = []
for temp in temperatures:
    text = generate_text(test_prompt, max_length=60, temperature=temp)
    continuation = text[len(test_prompt):]
    temp_results.append(len(set(continuation.split())))
    print(f"Temperature {temp}: {continuation[:100]}...")

# ──────────────────────────────────────────
# STEP 6: Visualize Temperature vs Vocabulary Diversity
# ──────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(temperatures, temp_results, marker='o', color='#7B1FA2',
         linewidth=2, markersize=8)
plt.fill_between(temperatures, temp_results, alpha=0.2, color='#7B1FA2')
plt.title('Temperature vs Vocabulary Diversity', fontsize=12, fontweight='bold')
plt.xlabel('Temperature')
plt.ylabel('Unique Words Used')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('temperature_analysis.png')
plt.show()
print("\nSaved: temperature_analysis.png")

# ──────────────────────────────────────────
# STEP 7: Generation Length Analysis
# ──────────────────────────────────────────
lengths = [r['length'] for r in results]
prompts_short = [f"Prompt {i+1}" for i in range(len(prompts))]

plt.figure(figsize=(10, 5))
plt.bar(prompts_short, lengths, color=['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0'])
plt.title('Generated Text Length per Prompt', fontsize=12, fontweight='bold')
plt.xlabel('Prompt')
plt.ylabel('Word Count')
for i, v in enumerate(lengths):
    plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('generation_lengths.png')
plt.show()
print("Saved: generation_lengths.png")

print("\n" + "=" * 60)
print("GPT-2 ANALYSIS COMPLETE!")
print("=" * 60)
print(f"Model    : GPT-2 (124M parameters)")
print(f"Prompts  : {len(prompts)} test prompts")
print(f"Charts   : 3 saved to LanguageModel/")
print("Files    : token_probabilities.png, temperature_analysis.png, generation_lengths.png")