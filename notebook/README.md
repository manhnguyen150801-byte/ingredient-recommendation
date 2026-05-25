# notebook/

This folder contains the main analysis notebook for the Ingredient Recommendation System.

---

## File

| File | Description |
|------|-------------|
| `group_submission.ipynb` | End-to-end notebook covering all 8 sections of the project |

---

## Notebook Structure

The notebook is divided into 8 numbered sections. Run them **in order** after a clean kernel restart.

### Section 1 — Data Loading
Loads `data/metadata.csv` (231,637 recipes) and `data/train.csv` (165,226 interactions). Parses the `nutrition` column into 7 named columns (`cal`, `protein`, `total_fat`, `sat_fat`, `total_carb`, `sugar`, `sodium`) and converts the string-encoded `ingredients` column into Python lists.

### Section 2 — Exploratory Data Analysis
- Top 30 most frequent ingredients (bar chart)
- Distribution of ingredients per recipe (histogram)
- Distribution of ratings in `train.csv` (bar chart with percentages)
- Calorie distribution across recipes (histogram)

All charts are saved to `outputs/charts/`.

### Section 3 — Train / Validation / Test Split
Defines "liked" recipes as those with `rating ≥ 4` (justification: 94% of all ratings are ≥ 4, making it the natural positive signal boundary). Builds a user–ingredient interaction table by exploding each liked recipe's ingredient list, then splits users 70/15/15 at the user level. Saves the three split files to `data/` for reuse in later sections.

### Section 4 — Top-N Ingredient Recommender
Builds a global ingredient co-occurrence matrix from all recipes in the metadata. For each user, scores unseen ingredients by summing co-occurrence counts with their known ingredients, with a global-popularity fallback. Tunes the list length K (5–25) on the validation set, then demonstrates Top-10 recommendations for 3 training users.

### Section 5 — Recipe Matching
Given a list of recommended ingredients, retrieves the best-matching recipes by overlap count and Jaccard similarity. Demonstrates end-to-end: user's Top-10 ingredients → 5 best matching recipes with names, matched ingredients, and calorie information.

### Section 6 — Ingredient Substitution
Represents each ingredient as its co-occurrence profile vector. Ranks substitutes for a query ingredient by cosine similarity between profiles. Tests three query ingredients: `butter`, `chicken`, and `flour`.

### Section 7 — Evaluation on Test Set
Reports Precision@K and Recall@K at K=5 and K=10 on the held-out test set. Also addresses the two optional objectives:
- **Coverage & Diversity**: Catalogue coverage (~0.27%) and average within-list pairwise dissimilarity (~0.25).
- **Calorie-aware Re-ranking**: Re-ranks candidates using a combined co-occurrence + calorie penalty score (α=0.4), reducing average calorie load by ~20 kcal.

### Section 8 — Conclusion
Markdown-only summary of capabilities, metrics, honest limitations, and future work.

---

## How to Run

Always launch Jupyter **from the project root** (one level above `notebook/`) so that relative paths resolve correctly:

```bash
cd ingredient-recommendation/   # project root
jupyter notebook notebook/group_submission.ipynb
```

Then: **Kernel → Restart & Run All**

The notebook expects:
- `data/metadata.csv` and `data/train.csv` to be present before running
- `outputs/charts/` directory to exist (created automatically by the notebook)

> **Note on raw data paths:** Sections 1 loads data using relative paths (`data/metadata.csv`). Make sure the data files are in the `data/` folder at the project root before running.
