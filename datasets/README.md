# Datasets for Basis Vectors in Persona Space

This directory contains datasets used for decomposing persona representations in LLMs.
Large data files are excluded from git; follow the instructions below to reproduce.

## 1. Anthropic Persona Evals

- **Source**: Perez et al. (2023), "Discovering Language Model Behaviors with Model-Written Evaluations"
- **URL**: https://github.com/anthropics/evals/tree/main/persona
- **Size**: ~12MB (135 JSONL files, 1000 items each)
- **Format**: JSONL with keys: `question`, `statement`, `label_confidence`, `answer_matching_behavior`, `answer_not_matching_behavior`
- **Content**: Binary agree/disagree statements for 135+ persona traits across categories:
  - Big Five personality (5 files): agreeableness, conscientiousness, extraversion, neuroticism, openness
  - Ethics (17 files): utilitarianism, deontology, virtue ethics, etc.
  - Politics (6 files): conservative, liberal, immigration, LGBTQ rights, gun rights, abortion
  - Plus: desires, unsafe behaviors, beliefs, religious views

### Download

```bash
cd datasets/anthropic-persona-evals
git clone --depth 1 --filter=blob:none --sparse https://github.com/anthropics/evals.git temp_evals
cd temp_evals && git sparse-checkout set persona
mv persona/* .. && cd .. && rm -rf temp_evals
```

## 2. PersonalityBench (from NPTI)

- **Source**: Huang et al. (2024), "NPTI: Neuron-based Personality Traits Induction in LLMs"
- **Paper**: https://arxiv.org/abs/2410.12327
- **Repository**: https://github.com/RUCAIBox/NPTI
- **Size**: ~27MB
- **Format**: JSONL files per Big Five trait
- **Content**:
  - `search/`: ~36,000 items per trait with keys: `BFI`, `tag`, `facet`, `topic`, `sign`, `question`
  - `test/`: ~90 situational questions per trait with key: `question`
  - `description.json`: IPIP-NEO-300 personality descriptions

### Download

```bash
cd datasets/personalitybench
git clone --depth 1 https://github.com/RUCAIBox/NPTI.git temp_npti
cp -r temp_npti/NPTI/dataset/* .
rm -rf temp_npti
```

## 3. PersonaLLM

- **Source**: Jiang et al. (2024), "PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits" (NAACL 2024)
- **Paper**: https://arxiv.org/abs/2305.02547
- **Repository**: https://github.com/hjian42/PersonaLLM
- **Size**: ~67MB
- **Format**: CSV scores + TXT stories
- **Content**:
  - `scores/`: BFI scores for GPT-3.5, GPT-4, LLaMA-2 (1600 rows each: 5 traits x 32 personality configs x 10 runs)
  - `text/gpt-4-0613/`: 320 stories (~800 words each) from GPT-4 with prescribed Big Five profiles
  - `text/gpt-3.5-turbo-0613/`: 320 stories from GPT-3.5
  - `text/human_stories/`: 2467 human-written stories for comparison
  - `text/llama-2/`: 320 stories from LLaMA-2

### Download

```bash
cd datasets/personallm
git clone --depth 1 https://github.com/hjian42/PersonaLLM.git .
```

## 4. BFI-44 (Big Five Inventory)

- **Source**: John, Donahue & Kentle (1991), "The Big Five Inventory -- Versions 4a and 54"
- **Size**: ~10KB
- **Format**: JSON and CSV
- **Content**: All 44 BFI items with trait assignment and reverse-scoring flags
  - Extraversion: 8 items (3 reverse)
  - Agreeableness: 9 items (4 reverse)
  - Conscientiousness: 9 items (4 reverse)
  - Neuroticism: 8 items (3 reverse)
  - Openness: 10 items (2 reverse)

### Files

- `bfi-44/bfi44_items.json`: Complete inventory with metadata
- `bfi-44/bfi44_items.csv`: Tabular format (id, text, trait, reverse)

These files are included in git (small size).

## Usage in Research

| Dataset | Use Case |
|---------|----------|
| Anthropic Persona Evals | Probing persona directions in activation space; contrastive pairs for steering vectors |
| PersonalityBench | Evaluating personality expression after intervention; situational personality tests |
| PersonaLLM | SVD analysis of personality-conditioned text; ground truth Big Five scores |
| BFI-44 | Reference questionnaire items; generating BFI-based prompts for LLMs |
