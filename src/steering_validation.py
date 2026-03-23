"""
Causal validation: Steer model with individual principal components
and measure whether behavior shifts meaningfully along the expected dimension.

For each of the top PCs:
1. Identify which personas load most strongly on that PC
2. Use held-out statements from those personas as test prompts
3. Steer the model by adding/subtracting the PC direction at the target layer
4. Measure whether the model's responses shift as predicted
"""

import json
import os
import random
import glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = "/workspaces/basis-vectors-persona-claude/results"
PLOTS_DIR = "/workspaces/basis-vectors-persona-claude/results/plots"
DATA_DIR = "/workspaces/basis-vectors-persona-claude/datasets/anthropic-persona-evals"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYER = 20  # Best layer from PCA analysis
N_TEST_STATEMENTS = 50
STEERING_STRENGTHS = [0, 5, 10, 20, 40]


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def get_pc_directions(layer_idx):
    """Load persona vectors, fit PCA, return PC directions in activation space."""
    vectors = np.load(os.path.join(RESULTS_DIR, f"persona_vectors_layer{layer_idx}.npy"))
    with open(os.path.join(RESULTS_DIR, "persona_names_final.json")) as f:
        names = json.load(f)

    X = vectors - vectors.mean(axis=0)
    pca = PCA(n_components=min(len(names), 50))
    X_pca = pca.fit_transform(X)

    # PC directions in the original activation space
    pc_directions = pca.components_  # shape: (n_components, hidden_dim)

    return pc_directions, pca, X_pca, names


def get_test_statements(persona_name, n=50):
    """Load held-out test statements for a persona."""
    fpath = os.path.join(DATA_DIR, f"{persona_name}.jsonl")
    if not os.path.exists(fpath):
        return [], []

    items = []
    with open(fpath) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("label_confidence", 0) >= 0.9:
                items.append(item)

    # Use items from the end (not used in training)
    random.shuffle(items)
    test_items = items[:n]

    questions = [item["question"] for item in test_items]
    expected = [item["answer_matching_behavior"].strip() for item in test_items]

    return questions, expected


def steer_and_measure(model, tokenizer, questions, expected_answers,
                       pc_direction, layer_idx, strengths, device="cuda"):
    """
    For each steering strength, inject the PC direction at the target layer
    and measure agreement rate with expected behavior.
    """
    pc_direction_tensor = torch.tensor(pc_direction, dtype=torch.float16, device=device)

    results = {}

    for alpha in strengths:
        correct = 0
        total = 0
        hooks = []

        def make_hook(alpha_val):
            def hook_fn(module, input, output):
                # output is a tuple; first element is the hidden states
                if isinstance(output, tuple):
                    hs = output[0]
                    hs = hs + alpha_val * pc_direction_tensor.unsqueeze(0).unsqueeze(0)
                    return (hs,) + output[1:]
                else:
                    return output + alpha_val * pc_direction_tensor.unsqueeze(0).unsqueeze(0)
            return hook_fn

        # Register hook at target layer
        if alpha != 0:
            hook = model.model.layers[layer_idx].register_forward_hook(make_hook(alpha))
            hooks.append(hook)

        for q, expected in zip(questions, expected_answers):
            inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=1.0,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # Check if response matches expected behavior
            if expected == "Yes" and ("yes" in response.lower() or response.lower().startswith("y")):
                correct += 1
            elif expected == "No" and ("no" in response.lower() or response.lower().startswith("n")):
                correct += 1
            total += 1

        # Remove hooks
        for h in hooks:
            h.remove()

        agreement_rate = correct / total if total > 0 else 0
        results[alpha] = {"agreement_rate": agreement_rate, "correct": correct, "total": total}
        print(f"    alpha={alpha:4d}: agreement={agreement_rate:.3f} ({correct}/{total})")

    return results


def main():
    print("Loading PC directions...")
    pc_directions, pca, X_pca, names = get_pc_directions(TARGET_LAYER)
    evr = pca.explained_variance_ratio_

    print(f"Top 5 PCs explain: {sum(evr[:5])*100:.1f}% variance")

    # For each of top 5 PCs, find the persona that loads most strongly (positive direction)
    # and steer with that PC to see if it increases agreement with that persona
    pc_test_configs = []
    for pc_idx in range(5):
        loadings = X_pca[:, pc_idx]
        # Positive direction: persona with highest loading
        pos_persona_idx = np.argmax(loadings)
        pos_persona = names[pos_persona_idx]
        # Negative direction: persona with lowest loading
        neg_persona_idx = np.argmin(loadings)
        neg_persona = names[neg_persona_idx]

        pc_test_configs.append({
            "pc_idx": pc_idx,
            "pos_persona": pos_persona,
            "neg_persona": neg_persona,
            "variance": float(evr[pc_idx]),
        })
        print(f"\nPC{pc_idx+1} ({evr[pc_idx]*100:.1f}%):")
        print(f"  + direction: {pos_persona}")
        print(f"  - direction: {neg_persona}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()

    all_steering_results = {}

    for config in pc_test_configs:
        pc_idx = config["pc_idx"]
        pos_persona = config["pos_persona"]
        neg_persona = config["neg_persona"]
        pc_dir = pc_directions[pc_idx]

        print(f"\n{'='*50}")
        print(f"PC{pc_idx+1}: Steering towards '{pos_persona}'")
        print(f"{'='*50}")

        questions, expected = get_test_statements(pos_persona)
        if len(questions) < 10:
            print(f"  Not enough test statements for {pos_persona}, skipping")
            continue

        questions = questions[:30]  # Limit for speed
        expected = expected[:30]

        print(f"  Testing with {len(questions)} statements")
        print(f"  Positive steering (should increase agreement):")
        pos_results = steer_and_measure(
            model, tokenizer, questions, expected,
            pc_dir, TARGET_LAYER, STEERING_STRENGTHS
        )

        print(f"  Negative steering (should decrease agreement):")
        neg_results = steer_and_measure(
            model, tokenizer, questions, expected,
            pc_dir, TARGET_LAYER, [-s for s in STEERING_STRENGTHS if s > 0]
        )

        all_steering_results[f"PC{pc_idx+1}"] = {
            "persona": pos_persona,
            "variance_pct": config["variance"],
            "positive_steering": {str(k): v for k, v in pos_results.items()},
            "negative_steering": {str(k): v for k, v in neg_results.items()},
        }

    # Save results
    with open(os.path.join(RESULTS_DIR, "steering_results.json"), "w") as f:
        json.dump(all_steering_results, f, indent=2)

    # Plot steering curves
    fig, axes = plt.subplots(1, len(all_steering_results), figsize=(5*len(all_steering_results), 4))
    if len(all_steering_results) == 1:
        axes = [axes]

    for i, (pc_name, data) in enumerate(all_steering_results.items()):
        ax = axes[i]

        # Combine positive and negative
        all_alphas = []
        all_rates = []

        for alpha_str, res in data["negative_steering"].items():
            all_alphas.append(float(alpha_str))
            all_rates.append(res["agreement_rate"])

        for alpha_str, res in data["positive_steering"].items():
            all_alphas.append(float(alpha_str))
            all_rates.append(res["agreement_rate"])

        # Sort by alpha
        sorted_pairs = sorted(zip(all_alphas, all_rates))
        alphas, rates = zip(*sorted_pairs)

        ax.plot(alphas, rates, 'o-', color='steelblue', linewidth=2)
        ax.axhline(y=data["positive_steering"]["0"]["agreement_rate"],
                   color='gray', linestyle='--', alpha=0.5, label='baseline')
        ax.set_xlabel("Steering Strength (α)")
        ax.set_ylabel("Agreement Rate")
        ax.set_title(f"{pc_name} ({data['variance_pct']*100:.1f}%)\n→ {data['persona'][:30]}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "steering_validation.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSteering results saved to {RESULTS_DIR}/steering_results.json")
    print(f"Plot saved to {PLOTS_DIR}/steering_validation.png")


if __name__ == "__main__":
    main()
