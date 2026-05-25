# slides/

This folder contains the project presentation delivered to the food.com consulting team.

---

## File

| File | Pages | Description |
|------|-------|-------------|
| `Slide.pdf` | 16 | Final project presentation covering brief, data, system design, results, and next steps |

---

## Slide Deck Outline

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Cover | Project title, course, team names |
| 2 | Agenda | Five sections: Project brief → Data & EDA → System design → Results → Limitations |
| 3 | What food.com Asked Us to Build | Three required deliverables + two optional objectives |
| 4 | Data at a Glance | Key statistics: 231K recipes, 55K users, 197K interactions, 14,942 unique ingredients |
| 5 | EDA — Ingredient Frequency & Calorie Profile | Salt appears in 85K+ recipes; median calorie = 313 kcal |
| 6 | Data Preparation | Rating ≥ 4 threshold → user–ingredient interaction table → 70/15/15 user-level split |
| 7–8 | System Design — Co-occurrence Model | How the co-occurrence matrix is built and used for scoring |
| 9–10 | Three Capabilities | Top-N recommendation, Recipe matching, Ingredient substitution with examples |
| 11–12 | Hyperparameter Tuning | Precision@K and Recall@K curves on the validation set; best K selection |
| 13 | Test Set Evaluation | Final metrics: P@5=0.58, P@10=0.50, R@10=0.12 |
| 14 | Optional Objectives | Coverage (0.27%), diversity analysis, calorie-aware re-ranking (−20 kcal) |
| 15 | Limitations | Popularity bias, cold-start, substitution ambiguity, calorie proxy issues |
| 16 | Next Steps | Matrix factorisation, diversity-forcing, multi-nutrient health score |

---

## Authors

- Duc Manh NGUYEN
- Minh Hoan TRAN
- Praveen PANDARINATHAM

IESEG School of Management — Recommendation Systems, 2nd Semester, 2026
