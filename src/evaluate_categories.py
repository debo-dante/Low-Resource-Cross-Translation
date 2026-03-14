import pandas as pd
import sacrebleu

# --- 1. Configuration ---
INPUT_FILE = "../data/evaluation/evaluation_results.csv"
REF_COL = "Ground_Truth_Kannada"      
PRED_COL = "Model_Prediction_Kannada" 

CATEGORIES = [
    "I. S-O-V Alignment & Basic Vocab",
    "II. Post-positions & Case Markers",
    "III. Honorifics & Pronoun Congruence",
    "IV. Complex Tenses & Conditionals",
    "V. Negation & Interrogatives"
]

def evaluate_by_category():
    try:
        df = pd.read_csv(INPUT_FILE)
        
        print("📊 CATEGORY-WISE PERFORMANCE BREAKDOWN")
        print("="*60)
        print(f"{'Category':<40} | {'BLEU':<6} | {'chrF++':<6}")
        print("-" * 60)

        # Loop through chunks of 20
        for i in range(5):
            start_idx = i * 20
            end_idx = start_idx + 20
            chunk = df.iloc[start_idx:end_idx]
            
            refs = [chunk[REF_COL].astype(str).tolist()]
            sys = chunk[PRED_COL].astype(str).tolist()
            
            bleu = sacrebleu.corpus_bleu(sys, refs)
            chrf = sacrebleu.corpus_chrf(sys, refs, word_order=2)
            
            print(f"{CATEGORIES[i]:<40} | {bleu.score:>6.2f} | {chrf.score:>6.2f}")
            
        print("="*60)
        
        all_refs = [df[REF_COL].astype(str).tolist()]
        all_sys = df[PRED_COL].astype(str).tolist()
        overall_bleu = sacrebleu.corpus_bleu(all_sys, all_refs)
        overall_chrf = sacrebleu.corpus_chrf(all_sys, all_refs, word_order=2)
        
        print(f"\n✅ OVERALL GOLD STANDARD METRICS:")
        print(f"Aggregate BLEU: {overall_bleu.score:.2f}")
        print(f"Aggregate chrF++: {overall_chrf.score:.2f}")
        
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found. Check path.")

if __name__ == "__main__":
    evaluate_by_category()
