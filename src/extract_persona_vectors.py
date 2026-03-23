"""
Extract persona vectors from LLM residual stream activations.

For each persona in the Anthropic persona evals dataset:
1. Select high-confidence contrastive statement pairs
2. Run statements through the model
3. Extract residual stream activations at specified layers
4. Compute persona vector = mean(matching) - mean(not_matching)
"""

import json
import os
import random
import glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Config
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/workspaces/basis-vectors-persona-claude/datasets/anthropic-persona-evals"
OUTPUT_DIR = "/workspaces/basis-vectors-persona-claude/results"
LAYERS_TO_EXTRACT = [8, 16, 20, 24, 28, 31]  # Early, mid, late layers
MAX_SAMPLES_PER_PERSONA = 100
MIN_CONFIDENCE = 0.85
BATCH_SIZE = 16


def load_persona_data(data_dir, min_confidence=0.85, max_samples=100):
    """Load and filter persona eval data from JSONL files."""
    persona_data = {}
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))

    for fpath in jsonl_files:
        persona_name = os.path.basename(fpath).replace(".jsonl", "")
        items = []
        with open(fpath) as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("label_confidence", 0) >= min_confidence:
                    items.append(item)

        if len(items) >= 20:  # Need minimum items for reliable vectors
            random.shuffle(items)
            persona_data[persona_name] = items[:max_samples]

    return persona_data


def prepare_contrastive_pairs(persona_items):
    """Split items into matching and not-matching behavior statements."""
    matching = []
    not_matching = []

    for item in persona_items:
        statement = item["statement"]
        answer_matching = item["answer_matching_behavior"].strip()

        if answer_matching == "Yes":
            matching.append(statement)
        else:
            not_matching.append(statement)

    return matching, not_matching


def extract_activations(model, tokenizer, statements, layers, batch_size=16, device="cuda"):
    """Extract residual stream activations at specified layers for a list of statements."""
    all_activations = {l: [] for l in layers}

    for i in range(0, len(statements), batch_size):
        batch = statements[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"]

        for layer_idx in layers:
            hs = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            # Use last non-padding token activation
            seq_lengths = attention_mask.sum(dim=1) - 1  # last valid token index
            batch_activations = []
            for b in range(hs.shape[0]):
                last_token_act = hs[b, seq_lengths[b]].float().cpu().numpy()
                batch_activations.append(last_token_act)
            all_activations[layer_idx].extend(batch_activations)

    # Convert to numpy arrays
    for layer_idx in layers:
        all_activations[layer_idx] = np.array(all_activations[layer_idx])

    return all_activations


def compute_persona_vector(matching_acts, not_matching_acts):
    """Compute persona direction vector via difference-in-means."""
    mean_matching = matching_acts.mean(axis=0)
    mean_not_matching = not_matching_acts.mean(axis=0)
    return mean_matching - mean_not_matching


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading persona data...")
    persona_data = load_persona_data(DATA_DIR, MIN_CONFIDENCE, MAX_SAMPLES_PER_PERSONA)
    print(f"Loaded {len(persona_data)} personas with sufficient high-confidence items")

    # Save persona list
    persona_names = sorted(persona_data.keys())
    with open(os.path.join(OUTPUT_DIR, "persona_names.json"), "w") as f:
        json.dump(persona_names, f, indent=2)

    # Print stats
    for name in persona_names[:5]:
        matching, not_matching = prepare_contrastive_pairs(persona_data[name])
        print(f"  {name}: {len(matching)} matching, {len(not_matching)} not-matching")
    print(f"  ... and {len(persona_names) - 5} more")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    # Get model info
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model: {n_layers} layers, hidden_dim={hidden_dim}")

    # Adjust layer indices if needed
    layers = [l for l in LAYERS_TO_EXTRACT if l < n_layers]
    print(f"Extracting from layers: {layers}")

    # Extract persona vectors
    persona_vectors = {l: {} for l in layers}
    persona_stats = {}

    for persona_name in tqdm(persona_names, desc="Extracting persona vectors"):
        matching_stmts, not_matching_stmts = prepare_contrastive_pairs(persona_data[persona_name])

        if len(matching_stmts) < 5 or len(not_matching_stmts) < 5:
            print(f"  Skipping {persona_name}: too few items ({len(matching_stmts)}/{len(not_matching_stmts)})")
            continue

        persona_stats[persona_name] = {
            "n_matching": len(matching_stmts),
            "n_not_matching": len(not_matching_stmts),
            "n_total": len(persona_data[persona_name]),
        }

        # Extract activations
        matching_acts = extract_activations(model, tokenizer, matching_stmts, layers, BATCH_SIZE)
        not_matching_acts = extract_activations(model, tokenizer, not_matching_stmts, layers, BATCH_SIZE)

        # Compute persona vector at each layer
        for layer_idx in layers:
            vec = compute_persona_vector(matching_acts[layer_idx], not_matching_acts[layer_idx])
            persona_vectors[layer_idx][persona_name] = vec

    # Save results
    print("\nSaving persona vectors...")
    for layer_idx in layers:
        names = sorted(persona_vectors[layer_idx].keys())
        vectors = np.array([persona_vectors[layer_idx][n] for n in names])
        np.save(os.path.join(OUTPUT_DIR, f"persona_vectors_layer{layer_idx}.npy"), vectors)
        print(f"  Layer {layer_idx}: {vectors.shape}")

    # Save the ordered name list (matching vector order)
    final_names = sorted(persona_vectors[layers[0]].keys())
    with open(os.path.join(OUTPUT_DIR, "persona_names_final.json"), "w") as f:
        json.dump(final_names, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "persona_stats.json"), "w") as f:
        json.dump(persona_stats, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "extraction_config.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "layers": layers,
            "max_samples_per_persona": MAX_SAMPLES_PER_PERSONA,
            "min_confidence": MIN_CONFIDENCE,
            "seed": SEED,
            "n_personas": len(final_names),
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
        }, f, indent=2)

    print(f"\nDone! Extracted vectors for {len(final_names)} personas across {len(layers)} layers.")
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
