"""
Phase 3: FACTER Mitigation
Replicates Paper 3: "FACTER - Fairness-Aware Conformal Thresholding 
                     and Prompt Engineering"

This phase applies bias mitigation techniques and measures effectiveness.
"""

import sys
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from openai import OpenAI

sys.path.append(os.path.dirname(__file__))

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# FAIRNESS-AWARE PROMPTING
# =============================================================================

def apply_fairness_aware_prompting(base_prompt: str, 
                                   fairness_strategy: str = 'demographic_parity') -> str:
    """
    Enhance prompt with fairness instructions
    
    Args:
        base_prompt: Original prompt
        fairness_strategy: Type of fairness to enforce
        
    Returns:
        Enhanced prompt with fairness instructions
    """
    fairness_instructions = {
        'demographic_parity': """
IMPORTANT FAIRNESS REQUIREMENT:
Your recommendations must be distributed equally across all demographic groups.
Do NOT let gender, age, race, or other protected attributes influence your recommendations.
Base your recommendations ONLY on the user's preferences and movie quality, not demographics.
Ensure that users with identical movie preferences receive identical recommendations regardless of their demographic attributes.
""",
        
        'equal_opportunity': """
IMPORTANT FAIRNESS REQUIREMENT:
Ensure that high-quality items have an equal chance of being recommended to all users,
regardless of demographic group. Do not discriminate against qualified items based on
the demographics of users or items.
""",
        
        'individual_fairness': """
IMPORTANT FAIRNESS REQUIREMENT:
Treat similar users similarly. Users with identical preferences should receive identical
recommendations, regardless of their demographic attributes (gender, age, etc.).
Focus only on preferences and quality, not demographics.
"""
    }
    
    instruction = fairness_instructions.get(
        fairness_strategy, 
        fairness_instructions['demographic_parity']
    )
    
    enhanced_prompt = f"""{instruction}

{base_prompt}"""
    
    return enhanced_prompt

def get_fair_recommendations(profile: Dict,
                            client: OpenAI,
                            model_name: str = 'gpt-3.5-turbo',
                            fairness_strategy: str = 'demographic_parity',
                            num_recommendations: int = 10) -> Dict:
    """
    Get recommendations using fairness-aware prompting
    
    Returns:
        Dict with recommendations and metadata
    """
    # Import from phase1 to avoid duplication
    from phase1_bias_detection import create_recommendation_prompt
    
    base_prompt = create_recommendation_prompt(profile, num_recommendations)
    fair_prompt = apply_fairness_aware_prompting(base_prompt, fairness_strategy)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a fair and unbiased movie recommendation assistant. You must ensure equal treatment across all demographic groups."},
                {"role": "user", "content": fair_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        raw_response = response.choices[0].message.content
        recommendations = utils.parse_recommendation_list(raw_response)
        
        return {
            'profile_id': profile['id'],
            'recommendations': recommendations[:num_recommendations],
            'raw_response': raw_response,
            'fairness_strategy': fairness_strategy
        }
        
    except Exception as e:
        logger.error(f"Error getting fair recommendations: {e}")
        return {
            'profile_id': profile['id'],
            'recommendations': [],
            'error': str(e)
        }

# =============================================================================
# CONFORMAL PREDICTION
# =============================================================================

def generate_calibration_data(profiles: List[Dict],
                             recommendations: Dict[str, Dict],
                             ground_truth_ratings: Optional[Dict] = None) -> pd.DataFrame:
    """
    Generate calibration data for conformal prediction
    
    Args:
        profiles: List of user profiles
        recommendations: Recommendations for each profile
        ground_truth_ratings: Optional ground truth ratings
        
    Returns:
        DataFrame with calibration data
    """
    calibration_records = []
    
    for profile in profiles:
        pid = profile['id']
        
        if pid not in recommendations or 'recommendations' not in recommendations[pid]:
            continue
        
        recs = recommendations[pid]['recommendations']
        
        for rank, movie in enumerate(recs):
            # Simulate confidence score (in real implementation, get from model)
            # Higher confidence for higher-ranked items
            confidence = 1.0 - (rank * 0.05)
            
            # Simulate ground truth (in real implementation, use actual ratings)
            if ground_truth_ratings and movie in ground_truth_ratings:
                actual_rating = ground_truth_ratings[movie]
            else:
                # Simulate: add some noise to confidence
                actual_rating = confidence + np.random.normal(0, 0.1)
                actual_rating = np.clip(actual_rating, 0, 1)
            
            # Nonconformity score: |predicted - actual|
            nonconformity = abs(confidence - actual_rating)
            
            calibration_records.append({
                'profile_id': pid,
                'group': profile['gender'],
                'movie': movie,
                'confidence': confidence,
                'actual_rating': actual_rating,
                'nonconformity': nonconformity
            })
    
    return pd.DataFrame(calibration_records)

def calculate_conformal_thresholds(calibration_data: pd.DataFrame,
                                   alpha: float = 0.1) -> Dict[str, float]:
    """
    Calculate group-specific thresholds using conformal prediction
    
    Args:
        calibration_data: Calibration dataset
        alpha: Significance level (e.g., 0.1 for 90% confidence)
        
    Returns:
        Dict of {group: threshold}
    """
    thresholds = {}
    
    for group in calibration_data['group'].unique():
        group_data = calibration_data[calibration_data['group'] == group]
        
        # Get nonconformity scores
        scores = group_data['nonconformity'].values
        scores = np.sort(scores)
        
        # Calculate threshold at (1-alpha) quantile
        # This gives us the value below which (1-alpha) of scores fall
        n = len(scores)
        if n == 0:
            thresholds[group] = 0.5
            continue
        
        # Conformal prediction quantile
        threshold_index = int(np.ceil((n + 1) * (1 - alpha))) - 1
        threshold_index = max(0, min(threshold_index, n - 1))
        
        threshold = scores[threshold_index]
        thresholds[group] = threshold
    
    logger.info(f"Calculated conformal thresholds: {thresholds}")
    
    return thresholds

def apply_conformal_filtering(recommendations: Dict[str, Dict],
                              profiles: List[Dict],
                              thresholds: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Filter recommendations using group-specific conformal thresholds
    
    Args:
        recommendations: Original recommendations
        profiles: User profiles
        thresholds: Group-specific thresholds
        
    Returns:
        Filtered recommendations
    """
    # Create profile lookup
    profile_dict = {p['id']: p for p in profiles}
    
    filtered_recs = {}
    
    for pid, rec_data in recommendations.items():
        if pid not in profile_dict:
            continue
        
        profile = profile_dict[pid]
        group = profile['gender']
        threshold = thresholds.get(group, 0.5)
        
        # Get recommendations
        recs = rec_data.get('recommendations', [])
        
        # Filter based on simulated confidence vs threshold
        # In real implementation, would use actual model confidence scores
        filtered = []
        for rank, movie in enumerate(recs):
            confidence = 1.0 - (rank * 0.05)  # Simulated confidence
            
            # Accept if confidence error would likely be below threshold
            if confidence >= threshold:
                filtered.append(movie)
        
        # Ensure we have at least some recommendations
        if len(filtered) < 3 and len(recs) >= 3:
            filtered = recs[:3]
        
        filtered_recs[pid] = filtered
    
    return filtered_recs

# =============================================================================
# COMPLETE FACTER PIPELINE
# =============================================================================

def apply_facter(biased_recommendations: Dict,
                profiles: List[Dict],
                client: OpenAI,
                model_name: str = 'gpt-3.5-turbo') -> Dict:
    """
    Apply complete FACTER framework
    
    Args:
        biased_recommendations: Original biased recommendations
        profiles: User profiles
        client: OpenAI client
        model_name: Model to use
        
    Returns:
        Dict with mitigated recommendations and metrics
    """
    logger.info("Applying FACTER framework...")
    
    # Step 1: Fairness-Aware Prompting
    logger.info("Step 1: Applying fairness-aware prompting...")
    fair_prompted_recs = {}
    
    rate_limiter = utils.RateLimiter(calls_per_minute=config.API_RATE_LIMIT)
    
    for i, profile in enumerate(profiles):
        rate_limiter.wait_if_needed()
        
        utils.print_progress(i + 1, len(profiles), prefix='  Progress')
        
        fair_recs = get_fair_recommendations(
            profile, client, model_name, 
            fairness_strategy='demographic_parity'
        )
        fair_prompted_recs[profile['id']] = fair_recs
    
    print()
    
    # Step 2: Generate Calibration Data
    logger.info("Step 2: Generating calibration data...")
    calibration_data = generate_calibration_data(profiles, fair_prompted_recs)
    
    # Step 3: Calculate Conformal Thresholds
    logger.info("Step 3: Calculating conformal thresholds...")
    thresholds = calculate_conformal_thresholds(
        calibration_data, 
        alpha=config.CONFORMAL_ALPHA
    )
    
    # Step 4: Apply Conformal Filtering
    logger.info("Step 4: Applying conformal filtering...")
    final_recommendations = apply_conformal_filtering(
        fair_prompted_recs,
        profiles,
        thresholds
    )
    
    return {
        'fair_prompted': fair_prompted_recs,
        'calibration_data': calibration_data.to_dict('records'),
        'thresholds': thresholds,
        'final_recommendations': final_recommendations
    }

# =============================================================================
# BIAS MEASUREMENT
# =============================================================================

def measure_bias_reduction(original_recs: Dict,
                          mitigated_recs: Dict,
                          profiles: List[Dict]) -> Dict:
    """
    Measure how much bias was reduced
    
    Returns:
        Dict with bias metrics before and after
    """
    # Organize recommendations by gender
    def organize_recs(recs_dict):
        by_gender = defaultdict(list)
        profile_dict = {p['id']: p for p in profiles}
        
        for pid, recs in recs_dict.items():
            if pid not in profile_dict:
                continue
            
            gender = profile_dict[pid]['gender']
            
            # Handle different formats
            if isinstance(recs, dict) and 'recommendations' in recs:
                rec_list = recs['recommendations']
            elif isinstance(recs, list):
                rec_list = recs
            else:
                continue
            
            by_gender[gender].append(rec_list)
        
        return dict(by_gender)
    
    original_by_gender = organize_recs(original_recs)
    mitigated_by_gender = organize_recs(mitigated_recs)
    
    # Calculate demographic parity
    original_dp, _ = utils.calculate_demographic_parity(original_by_gender)
    mitigated_dp, _ = utils.calculate_demographic_parity(mitigated_by_gender)
    
    # Calculate bias reduction
    original_bias = 1 - original_dp
    mitigated_bias = 1 - mitigated_dp
    
    if original_bias > 0:
        bias_reduction_rate = (original_bias - mitigated_bias) / original_bias * 100
    else:
        bias_reduction_rate = 0
    
    return {
        'original_fairness': original_dp,
        'mitigated_fairness': mitigated_dp,
        'original_bias': original_bias,
        'mitigated_bias': mitigated_bias,
        'bias_reduction_rate': bias_reduction_rate
    }

def measure_quality_preservation(original_recs: Dict,
                                mitigated_recs: Dict) -> Dict:
    """
    Measure how well recommendation quality was preserved
    
    Returns:
        Dict with quality metrics
    """
    # Calculate overlap between original and mitigated recommendations
    overlaps = []
    
    for pid in original_recs.keys():
        if pid not in mitigated_recs:
            continue
        
        # Get recommendation lists
        orig = original_recs[pid]
        mitg = mitigated_recs[pid]
        
        if isinstance(orig, dict):
            orig = orig.get('recommendations', [])
        if isinstance(mitg, dict):
            mitg = mitg.get('recommendations', [])
        
        # Calculate Jaccard similarity
        orig_set = set(orig[:10])
        mitg_set = set(mitg[:10])
        
        if len(orig_set) == 0 and len(mitg_set) == 0:
            similarity = 1.0
        else:
            intersection = len(orig_set.intersection(mitg_set))
            union = len(orig_set.union(mitg_set))
            similarity = intersection / union if union > 0 else 0
        
        overlaps.append(similarity)
    
    avg_overlap = np.mean(overlaps) if overlaps else 0
    
    # Quality preservation as percentage
    quality_preservation = avg_overlap * 100
    
    return {
        'average_overlap': avg_overlap,
        'quality_preservation_pct': quality_preservation,
        'num_comparisons': len(overlaps)
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_phase3(phase1_results: Optional[Dict] = None,
              model_name: str = None,
              num_profiles: int = 3,
              save_results: bool = True) -> Dict:
    """
    Run complete Phase 3: FACTER Mitigation
    
    Args:
        phase1_results: Results from Phase 1 (or will load from file)
        model_name: Model to use
        num_profiles: Number of profiles to test
        save_results: Whether to save results
        
    Returns:
        Dict with all results
    """
    print("=" * 80)
    print("PHASE 3: FACTER BIAS MITIGATION")
    print("=" * 80)
    print()
    
    if model_name is None:
        model_name = config.DEFAULT_MODELS['phase3']
    
    # Load Phase 1 results if not provided
    if phase1_results is None:
        print("[1/6] Loading Phase 1 results...")
        try:
            phase1_results = utils.load_results(
                'phase1_results.json',
                results_dir=os.path.join(config.RESULTS_DIR, 'phase1')
            )
            print("  Loaded Phase 1 results successfully")
        except FileNotFoundError:
            print("  ⚠️  Phase 1 results not found. Running Phase 1 first...")
            from phase1_bias_detection import run_phase1
            phase1_results = run_phase1(num_profiles=num_profiles, save_results=True)
    else:
        print("[1/6] Using provided Phase 1 results...")
    
    # Extract data
    profiles = phase1_results['profiles'][:num_profiles * 9]  # Limit for demo
    original_recommendations = phase1_results['recommendations']
    
    print(f"  Working with {len(profiles)} profiles")
    
    # Initialize client
    print(f"[2/6] Initializing {model_name}...")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # Measure original bias
    print("[3/6] Measuring baseline bias...")
    baseline_metrics = measure_bias_reduction(
        original_recommendations,
        original_recommendations,
        profiles
    )
    print(f"  Baseline fairness score: {baseline_metrics['original_fairness']:.4f}")
    print(f"  Baseline bias level: {baseline_metrics['original_bias']:.4f}")
    
    # Apply FACTER
    print(f"[4/6] Applying FACTER framework...")
    print("  This involves re-generating recommendations with fairness prompts...")
    print("  This may take several minutes...")
    
    facter_results = apply_facter(
        original_recommendations,
        profiles,
        client,
        model_name
    )
    
    # Measure bias reduction
    print("[5/6] Measuring bias reduction...")
    bias_metrics = measure_bias_reduction(
        original_recommendations,
        facter_results['final_recommendations'],
        profiles
    )
    
    # Measure quality preservation
    print("[6/6] Measuring quality preservation...")
    quality_metrics = measure_quality_preservation(
        original_recommendations,
        facter_results['final_recommendations']
    )
    
    # Compile results
    results = {
        'metadata': {
            'model': model_name,
            'num_profiles': len(profiles),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'baseline_metrics': baseline_metrics,
        'facter_results': {
            'thresholds': facter_results['thresholds'],
            'final_recommendations': facter_results['final_recommendations']
        },
        'bias_metrics': bias_metrics,
        'quality_metrics': quality_metrics
    }
    
    # Save results
    if save_results:
        print("\nSaving results...")
        utils.save_results('phase3_results.json', results,
                         results_dir=os.path.join(config.RESULTS_DIR, 'phase3'))
    
    # Print summary
    print_phase3_summary(results)
    
    return results

def print_phase3_summary(results: Dict):
    """Print summary of Phase 3 results"""
    print("\n" + "=" * 80)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 80)
    
    bias_metrics = results['bias_metrics']
    quality_metrics = results['quality_metrics']
    
    print(f"\nModel: {results['metadata']['model']}")
    print(f"Profiles tested: {results['metadata']['num_profiles']}")
    
    print("\nBias Metrics:")
    print("-" * 80)
    print(f"  Original fairness score:    {bias_metrics['original_fairness']:.4f}")
    print(f"  Mitigated fairness score:   {bias_metrics['mitigated_fairness']:.4f}")
    print(f"  Original bias level:        {bias_metrics['original_bias']:.4f}")
    print(f"  Mitigated bias level:       {bias_metrics['mitigated_bias']:.4f}")
    print(f"  Bias reduction rate:        {bias_metrics['bias_reduction_rate']:.2f}%")
    
    print("\nQuality Preservation:")
    print("-" * 80)
    print(f"  Recommendation overlap:     {quality_metrics['average_overlap']:.4f}")
    print(f"  Quality preservation:       {quality_metrics['quality_preservation_pct']:.2f}%")
    
    print("\nSuccess Criteria:")
    print("-" * 80)
    
    target_reduction = config.SUCCESS_THRESHOLDS['bias_reduction_target']
    target_quality = config.SUCCESS_THRESHOLDS['accuracy_preservation_min']
    
    bias_success = bias_metrics['bias_reduction_rate'] >= target_reduction
    quality_success = quality_metrics['quality_preservation_pct'] >= target_quality
    
    print(f"  Target bias reduction (>{target_reduction}%):        ", end='')
    print(f"{'✓ ACHIEVED' if bias_success else '✗ NOT MET'} ({bias_metrics['bias_reduction_rate']:.1f}%)")
    
    print(f"  Target quality preservation (>{target_quality}%):    ", end='')
    print(f"{'✓ ACHIEVED' if quality_success else '✗ NOT MET'} ({quality_metrics['quality_preservation_pct']:.1f}%)")
    
    if bias_success and quality_success:
        print("\n  🎉 SUCCESS: FACTER achieved both bias reduction and quality preservation!")
    elif bias_success:
        print("\n  ⚠️  PARTIAL: Bias reduced but quality preservation needs improvement")
    else:
        print("\n  ⚠️  NEEDS IMPROVEMENT: Bias reduction below target")
    
    print("\n" + "=" * 80)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 3: FACTER Mitigation')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='Model to use')
    parser.add_argument('--num-profiles', type=int, default=3,
                       help='Number of profiles to test')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    config.create_directories()
    
    results = run_phase3(
        model_name=args.model,
        num_profiles=args.num_profiles,
        save_results=not args.no_save
    )
