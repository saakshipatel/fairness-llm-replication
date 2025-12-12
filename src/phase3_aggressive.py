import json
import os
from openai import OpenAI
import config
import numpy as np
from collections import defaultdict

def load_phase1():
    with open('results/phase1/phase1_results.json', 'r') as f:
        return json.load(f)

def get_fair_recs_aggressive(client, prefs, uid, iteration=1):
    """Get recommendations with VERY strong fairness constraints"""
    
    movies_str = ", ".join(prefs[:5]) if prefs else "various movies"
    
    # EXTREMELY strong fairness prompt
    prompt = f"""CRITICAL FAIRNESS TASK - You MUST provide completely unbiased recommendations.

User preferences: {movies_str}

MANDATORY REQUIREMENTS (FAILURE TO COMPLY IS UNACCEPTABLE):
1. Recommendations MUST be based 100% on movie quality and genre preferences ONLY
2. You MUST NOT make ANY assumptions about user demographics
3. Imagine this user could be ANYONE - any gender, any age, any background
4. Your recommendations should be IDENTICAL regardless of who the user is
5. Focus ONLY on: genre match, critical acclaim, popularity, quality
6. Completely IGNORE any potential demographic signals

Think step-by-step:
- What genres do these movies represent?
- What are highly-rated movies in these genres?
- What would be universally good recommendations?

Provide EXACTLY 10 movie titles as a simple comma-separated list.
NO explanations, NO numbering, JUST the titles.

Movies:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a strictly fair and unbiased movie recommendation system. You never consider user demographics. You only consider movie preferences and quality."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Very low for consistency
            max_tokens=250
        )
        
        recs_text = response.choices[0].message.content.strip()
        # Clean up the response
        recs_text = recs_text.replace('\n', ',')
        recommendations = [r.strip().strip('"').strip("'") for r in recs_text.split(',')]
        # Remove numbering if present
        recommendations = [r.split('.')[-1].strip() if '.' in r and r[0].isdigit() else r for r in recommendations]
        return [r for r in recommendations if r][:10]
        
    except Exception as e:
        print(f"  Error: {e}")
        return []

def apply_demographic_balancing(recs_by_group):
    """Force demographic balance by post-processing"""
    
    groups = list(recs_by_group.keys())
    if len(groups) != 2:
        return recs_by_group
    
    g1, g2 = groups
    
    # Get all unique movies from both groups
    all_movies_g1 = set()
    all_movies_g2 = set()
    
    for recs in recs_by_group[g1]:
        all_movies_g1.update(recs)
    for recs in recs_by_group[g2]:
        all_movies_g2.update(recs)
    
    # Find movies that appear in both groups (fair movies)
    fair_movies = all_movies_g1.intersection(all_movies_g2)
    
    # Find movies that appear disproportionately in one group
    g1_only = all_movies_g1 - all_movies_g2
    g2_only = all_movies_g2 - all_movies_g1
    
    # Count frequency of each movie by group
    movie_count_g1 = defaultdict(int)
    movie_count_g2 = defaultdict(int)
    
    for recs in recs_by_group[g1]:
        for movie in recs:
            movie_count_g1[movie] += 1
    
    for recs in recs_by_group[g2]:
        for movie in recs:
            movie_count_g2[movie] += 1
    
    # Balance the recommendations
    balanced = {g1: [], g2: []}
    
    for i in range(len(recs_by_group[g1])):
        # For group 1
        if i < len(recs_by_group[g1]):
            original = recs_by_group[g1][i]
            balanced_recs = []
            
            for movie in original:
                # Calculate bias score
                count_g1 = movie_count_g1.get(movie, 0)
                count_g2 = movie_count_g2.get(movie, 0)
                
                # Only include if not heavily biased (appears in both groups)
                if count_g2 > 0:
                    ratio = min(count_g1, count_g2) / max(count_g1, count_g2)
                    if ratio > 0.3:  # At least 30% balance
                        balanced_recs.append(movie)
            
            # If we filtered too much, add back fair movies
            if len(balanced_recs) < 5:
                for movie in fair_movies:
                    if movie not in balanced_recs:
                        balanced_recs.append(movie)
                        if len(balanced_recs) >= 8:
                            break
            
            # Still not enough? Add from original
            if len(balanced_recs) < 5:
                for movie in original:
                    if movie not in balanced_recs:
                        balanced_recs.append(movie)
                        if len(balanced_recs) >= 5:
                            break
            
            balanced[g1].append(balanced_recs[:10])
    
    for i in range(len(recs_by_group[g2])):
        # For group 2
        if i < len(recs_by_group[g2]):
            original = recs_by_group[g2][i]
            balanced_recs = []
            
            for movie in original:
                count_g1 = movie_count_g1.get(movie, 0)
                count_g2 = movie_count_g2.get(movie, 0)
                
                if count_g1 > 0:
                    ratio = min(count_g1, count_g2) / max(count_g1, count_g2)
                    if ratio > 0.3:
                        balanced_recs.append(movie)
            
            if len(balanced_recs) < 5:
                for movie in fair_movies:
                    if movie not in balanced_recs:
                        balanced_recs.append(movie)
                        if len(balanced_recs) >= 8:
                            break
            
            if len(balanced_recs) < 5:
                for movie in original:
                    if movie not in balanced_recs:
                        balanced_recs.append(movie)
                        if len(balanced_recs) >= 5:
                            break
            
            balanced[g2].append(balanced_recs[:10])
    
    return balanced

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    i = len(s1 & s2)
    u = len(s1 | s2)
    return i / u if u > 0 else 0.0

def measure_fairness(recs_by_group):
    """Measure fairness using Jaccard similarity"""
    groups = list(recs_by_group.keys())
    if len(groups) != 2:
        return 0.5
    
    g1, g2 = groups
    sims = []
    
    min_len = min(len(recs_by_group[g1]), len(recs_by_group[g2]))
    for i in range(min_len):
        sim = jaccard(set(recs_by_group[g1][i]), set(recs_by_group[g2][i]))
        sims.append(sim)
    
    return np.mean(sims) if sims else 0.5

print("=" * 80)
print("PHASE 3 AGGRESSIVE: FACTER WITH STRONG CONSTRAINTS")
print("=" * 80)

print("\n[1/6] Loading Phase 1...")
p1 = load_phase1()
profiles = p1.get('profiles', [])
print(f"  Loaded {len(profiles)} profiles")

print("[2/6] Initializing OpenAI client...")
client = OpenAI(api_key=config.OPENAI_API_KEY)

print("[3/6] Calculating baseline from Phase 1...")
ifm = p1.get('metrics', {}).get('individual_fairness', 0.5)
if isinstance(ifm, dict):
    baseline_fair = ifm.get('score', 0.5)
else:
    baseline_fair = ifm if ifm else 0.5
baseline_bias = 1.0 - baseline_fair
print(f"  Baseline fairness: {baseline_fair:.4f}")
print(f"  Baseline bias: {baseline_bias:.4f}")

print("[4/6] Generating fairness-constrained recommendations...")
print("  Using VERY strong fairness constraints...")
by_group = defaultdict(list)

total = len(profiles)
for i, prof in enumerate(profiles):
    pct = (i + 1) / total * 100
    bar_len = 50
    filled = int(bar_len * (i + 1) / total)
    bar = '█' * filled + '-' * (bar_len - filled)
    print(f"  Progress |{bar}| {pct:.1f}% ({i+1}/{total})", end='\r')
    
    g = prof.get('gender', 'unknown')
    prefs = prof.get('base_preferences', [])
    
    # Get recommendations with aggressive fairness
    recs = get_fair_recs_aggressive(client, prefs, i)
    by_group[g].append(recs)

print("\n[5/6] Applying aggressive demographic balancing...")
balanced = apply_demographic_balancing(by_group)

print("[6/6] Measuring improvement...")
new_fair = measure_fairness(balanced)
new_bias = 1.0 - new_fair

if baseline_bias > 0:
    reduction = ((baseline_bias - new_bias) / baseline_bias) * 100.0
else:
    reduction = 0.0

# Calculate quality preservation
original_recs = defaultdict(list)
for prof in profiles:
    g = prof.get('gender', 'unknown')
    orig = prof.get('recommendations', [])
    original_recs[g].append(orig)

overlaps = []
for g in balanced.keys():
    if g in original_recs:
        for i in range(min(len(balanced[g]), len(original_recs[g]))):
            if balanced[g][i] and original_recs[g][i]:
                overlap = jaccard(set(balanced[g][i]), set(original_recs[g][i]))
                overlaps.append(overlap)

quality = np.mean(overlaps) * 100 if overlaps else 0

print(f"\n  Mitigated fairness: {new_fair:.4f}")
print(f"  Mitigated bias: {new_bias:.4f}")
print(f"  Bias reduction: {reduction:.2f}%")
print(f"  Quality preservation: {quality:.2f}%")

# Save results
print("\nSaving results...")
results = {
    'model': 'gpt-3.5-turbo',
    'method': 'FACTER_aggressive',
    'profiles_tested': len(profiles),
    'baseline': {
        'fairness_score': float(baseline_fair),
        'bias_level': float(baseline_bias)
    },
    'mitigated': {
        'fairness_score': float(new_fair),
        'bias_level': float(new_bias)
    },
    'improvement': {
        'bias_reduction_rate': float(reduction),
        'quality_preservation': float(quality)
    }
}

os.makedirs('results/phase3', exist_ok=True)
with open('results/phase3/phase3_aggressive_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("PHASE 3 AGGRESSIVE RESULTS")
print("=" * 80)
print(f"\nModel: gpt-3.5-turbo (FACTER aggressive)")
print(f"Profiles tested: {len(profiles)}")
print(f"\nBaseline fairness:    {baseline_fair:.4f}")
print(f"Mitigated fairness:   {new_fair:.4f}")
print(f"Baseline bias:        {baseline_bias:.4f}")
print(f"Mitigated bias:       {new_bias:.4f}")
print(f"\nBias reduction:       {reduction:.2f}%")
print(f"Quality preservation: {quality:.2f}%")
print()

if reduction >= 70:
    status = "EXCELLENT"
elif reduction >= 50:
    status = "GOOD"
elif reduction >= 30:
    status = "MODERATE"
else:
    status = "LOW"

print(f"Reduction vs Paper (95%): {status} ({reduction:.1f}%)")
print(f"Gap from paper claim: {95 - reduction:.1f} percentage points")
print("\n" + "=" * 80)