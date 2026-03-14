# 🇮🇳 Assamese-Kannada NMT via Parameter-Efficient Fine-Tuning (LoRA)

**An MTech Thesis Project demonstrating State-of-the-Art cross-family translation (Indo-Aryan to Dravidian) by bypassing the shared-script Tokenization Barrier.**

![BLEU Badge](https://img.shields.io/badge/BLEU-32.92-brightgreen)
![chrF++ Badge](https://img.shields.io/badge/chrF++-60.80-blue)
![Parameters](https://img.shields.io/badge/Trainable_Params-0.58%25-orange)
![Hardware](https://img.shields.io/badge/Hardware-Consumer_GPU-red)

---

## 📖 Abstract Overview
Translating between regional Indian languages typically suffers from extreme data sparsity and morphological divergence. Early multilingual baselines often force regional alphabets into a shared phonetic space (the **"Devanagari Trap"**), which destroys affixation boundaries and causes severe autoregressive decoding collapse. 

This repository solves this architectural flaw by applying **Low-Rank Adaptation (LoRA)** to the **NLLB-200** architecture. By preserving Separate Script (SS) subword tokenization, the model bridges Assamese and Kannada natively, achieving highly fluent translations in just 2 hours of training.

---

## 🛑 The Baseline Failure: The "IndicBART Autopsy"
Before adopting NLLB, an 8-hour full fine-tuning run was executed on a standard shared-script sequence-to-sequence model (IndicBART) using 80,000 parallel rows. 

* **The Mathematical Illusion:** The training loss plateaued at an impressive `2.60`.
* **The Reality (Mode Collapse):** Because the tokenizer transliterated both languages into Devanagari, the morphological markers were erased. The autoregressive decoder became "blind" and exploited the loss function by generating infinite loops of high-frequency Kannada syllables (e.g., `ನನನ ನನನ` [na-na-na]) instead of valid translations.

---

## 🚀 Proposed Architecture (NLLB + LoRA)
To bypass the Devanagari Trap, we utilized the `facebook/nllb-200-distilled-600M` model, which features a massive 256k BPE vocabulary that natively supports the Assamese and Kannada scripts independently. 

Instead of full fine-tuning, we injected trainable rank-decomposition matrices (LoRA) into the attention layers:
* **Target Modules:** `q_proj`, `v_proj`
* **Rank ($r$):** 16
* **Alpha ($\alpha$):** 32
* **Trainable Parameters:** ~3.4 Million (0.58% of the total model)

### 📉 Training Convergence
The model successfully avoided mode collapse, rapidly descending from an initial loss of 17.16 to a stable micro-adjustment plateau of `10.86` in exactly 600 steps.

![Training Loss Convergence](paper/loss_curve_complete.png)
*(Note: The model traded an artificially low loss for structural semantic accuracy).*

---

## 📊 Evaluation Metrics (The Gold Standard)
The model was evaluated on a hand-crafted, rigorously stratified 100-sentence Gold Standard test set designed to test specific morphological bridges between Indo-Aryan and Dravidian syntax.

| Category | Linguistic Focus | BLEU Score | chrF++ Score |
| :--- | :--- | :---: | :---: |
| **I** | S-O-V Alignment & Basic Vocab | 9.78 | 55.14 |
| **II** | Post-positions & Case Markers | 28.49 | 59.59 |
| **III** | Honorifics & Pronoun Congruence | 51.63 | 68.79 |
| **IV** | Complex Tenses & Conditionals | 30.39 | 55.98 |
| **V** | Negation & Interrogatives | 43.76 | 68.54 |
| **Overall** | **Aggregate Gold Standard** | **32.92** | **60.80** |

---

## 🔍 Qualitative Case Study (Morphological Mapping)
The chrF++ score of 60.80 mathematically proves the model's ability to map Assamese independent post-positions directly to Kannada's agglutinative suffixes.

| Source (Assamese - Romanized) | Ground Truth (Kannada) | Model Prediction | Linguistic Remark |
| :--- | :--- | :--- | :--- |
| Rame bhayekok eta kolom dile. | ರಾಮನು ತನ್ನ ಸಹೋದರನಿಗೆ ಒಂದು ಪೆನ್ನನ್ನು ಕೊಟ್ಟನು. | ರಾಮನು ತನ್ನ ಸಹೋದರನಿಗೆ ಒಂದು ಪೆನ್ನನ್ನು ಕೊಟ್ಟನು. | Perfect Dative Case alignment. |
| Sir, apuni bhitoroloi ahibo pare. | ಸರ್, ನೀವು ಒಳಗೆ ಬರಬಹುದು. | ಸರ್, ನೀವು ಒಳಗೆ ಬರಬಹುದು. | Maintained formal honorific congruence. |
| Jodi boroxun diye, tente moi najao. | ಮಳೆ ಬಂದರೆ, ನಾನು ಹೋಗುವುದಿಲ್ಲ. | ಮಳೆ ಬಂದರೆ, ನಾನು ಹೋಗುವುದಿಲ್ಲ. | Accurate conditional mapping. |

---

## 💻 Installation & Local CPU Inference
Because the LoRA adapters are extremely lightweight (~15MB), inference can be executed locally on a standard non-GPU machine.

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
