# Fairness in LLM Recommendations - Replication Study

**Team :** Group 3 - Saakshi, Monu, Om  
**Course:** CS 516  
**Fall 2025**

## Project Overview

Replicated three research papers about fairness in AI recommendation systems to test if their claims actually hold up. Turns out ChatGPT is pretty biased when recommending movies, but we can reduce most of that bias using the techniques from the papers.

## The Papers

1. Zhang et al. (SIGIR 2023) - Detecting bias in ChatGPT recommendations - https://arxiv.org/pdf/2305.07609
2. Wang et al. (ACL 2024) - Comparing GPT-3.5 vs GPT-4 for fairness - https://arxiv.org/pdf/2404.03192
3. Chen et al. (2024) - FACTER framework for reducing bias - https://arxiv.org/pdf/2502.02966

## Results

### Phase 1: Does ChatGPT Have Bias?
**Yes, definitely.**

- Demographic Parity: 0.68 (paper also got 0.68 - exact match!)
- Individual Fairness: 0.47 (below the 0.85 threshold for "fair")
- Equal Opportunity: 1.00 (this one was fine)

We created fake user profiles with identical movie preferences but different demographics. ChatGPT gave different recommendations based on gender/age even though the preferences were the same.

### Phase 2: Is GPT-4 Better Than GPT-3.5?
**Yes, but not by much.**

- GPT-4: 0.89 fairness score
- GPT-3.5: 0.85 fairness score  
- About 5% improvement

So newer models help, but they're still not perfect.

### Phase 3: Can We Fix It?
**Mostly yes.**

Using the FACTER approach (fairness-aware prompts + demographic balancing):
- Achieved: 89.7% bias reduction
- Paper claimed: 95% reduction
- We got within 5.3% of their target

Pretty close! Some implementation details from the paper weren't super clear, so that's probably why we didn't hit exactly 95%.


## How to Run This

### What You Need
- Python 3.8+
- OpenAI API key (costs ~$2 total for all experiments)
- MovieLens 1M dataset

### Setup

```bash
# Install packages
pip install -r requirements.txt

# Get the dataset
# Go to: https://grouplens.org/datasets/movielens/1m/
# Download ml-1m.zip and extract to data/ml-1m/

# Add your API key
# Copy src/config.py.example to src/config.py
# Add your OpenAI API key in that file
```

### Run Experiments

```bash
# Phase 1 - Detect bias (~15 min, ~$0.30)
python src/phase1_bias_detection.py

# Phase 2 - Compare models (~20 min, ~$0.50)
python src/phase2_cross_model_eval.py

# Phase 3 - Test mitigation (~25 min, ~$0.60)
python src/phase3_final.py
```

Results get saved as JSON files in the `results/` folder.

## Code Overview

### Phase 1: Bias Detection (phase1_bias_detection.py)
**What it does:** Tests if ChatGPT gives biased recommendations based on demographics

**Key functions:**
- `load_movielens_data()` - Loads and parses MovieLens dataset
- `create_synthetic_profiles()` - Generates 45 user profile pairs with identical preferences but different demographics
- `get_chatgpt_recommendations()` - Calls OpenAI API to get recommendations for each profile
- `calculate_demographic_parity()` - Measures bias between demographic groups
- `calculate_individual_fairness()` - Compares recommendations for similar users
- `calculate_equal_opportunity()` - Measures fair access to quality recommendations

**Output:** JSON file with three fairness metrics (DP, IF, EO)

### Phase 2: Cross-Model Evaluation (phase2_cross_model_eval.py)
**What it does:** Compares GPT-3.5 and GPT-4 fairness on ranking tasks

**Key functions:**
- `create_ranking_tasks()` - Generates synthetic ranking scenarios
- `evaluate_listwise()` - Tests model on ranking multiple items at once
- `evaluate_pairwise()` - Tests model on comparing pairs of items
- `calculate_exposure_ratio()` - Measures how fairly items are ranked across groups
- `compare_models()` - Runs both models and compares fairness scores

**Output:** JSON file with fairness scores for both models

### Phase 3: FACTER Mitigation (phase3_facter_mitigation.py)
**What it does:** Applies bias mitigation techniques and measures improvement

**Key functions:**
- `apply_fairness_aware_prompting()` - Adds fairness constraints to prompts
- `get_fair_recommendations()` - Gets recommendations with fairness instructions
- `apply_demographic_balancing()` - Post-processes results to balance across groups
- `filter_balanced_movies()` - Identifies movies that appear fairly across demographics
- `measure_fairness_improvement()` - Calculates before/after bias metrics
- `generate_mitigation_report()` - Creates summary of results

**Output:** JSON file with baseline fairness, mitigated fairness, and reduction percentage

### Utility Functions (utils.py)
- Data loading helpers
- JSON file I/O
- Metric calculation utilities
- Logging setup

**Key insights:**
- Bias in LLMs is real and measurable
- Bigger models (GPT-4) are better but still need fixing
- You can reduce most bias without retraining the model
- There's a tradeoff between fairness and keeping original recommendations

## References

1. Zhang, Y., et al. (2023). "Is ChatGPT Fair for Recommendation?" SIGIR 2023.
2. Wang, W., et al. (2024). "Do Large Language Models Rank Fairly?" ACL 2024.
3. Chen, J., et al. (2024). "FACTER: Fairness-Aware Mitigation Framework."

*Built with too much coffee and occasional frustration with API timeouts*
