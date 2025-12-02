"""
Utility functions for Fairness in LLM Replication Study
Shared helper functions used across all phases
"""

import json
import pickle
import os
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FILE I/O OPERATIONS
# =============================================================================

def save_results(filename: str, data: Any, results_dir: str = '../results'):
    """
    Save results to JSON file
    
    Args:
        filename: Name of file to save
        data: Data to save (will be JSON serialized)
        results_dir: Directory to save results in
    """
    filepath = os.path.join(results_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")
        raise

def load_results(filename: str, results_dir: str = '../results') -> Any:
    """
    Load results from JSON file
    
    Args:
        filename: Name of file to load
        results_dir: Directory to load results from
        
    Returns:
        Loaded data
    """
    filepath = os.path.join(results_dir, filename)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Results loaded from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {e}")
        raise

def save_pickle(filename: str, data: Any, results_dir: str = '../results'):
    """Save data using pickle (for complex Python objects)"""
    filepath = os.path.join(results_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Pickle saved to {filepath}")

def load_pickle(filename: str, results_dir: str = '../results') -> Any:
    """Load data from pickle file"""
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Pickle loaded from {filepath}")
    return data

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def calculate_kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Calculate KL divergence between two probability distributions
    
    Args:
        p: First distribution (dict of {category: probability})
        q: Second distribution (dict of {category: probability})
        
    Returns:
        KL divergence value
    """
    # Get all unique keys
    all_keys = set(list(p.keys()) + list(q.keys()))
    
    # Create arrays with small epsilon for missing values
    epsilon = 1e-10
    p_array = np.array([p.get(k, epsilon) for k in all_keys])
    q_array = np.array([q.get(k, epsilon) for k in all_keys])
    
    # Normalize to ensure they sum to 1
    p_array = p_array / p_array.sum()
    q_array = q_array / q_array.sum()
    
    # Calculate KL divergence
    return entropy(p_array, q_array)

def calculate_js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Calculate Jensen-Shannon divergence (symmetric version of KL divergence)
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        JS divergence value (0 = identical, 1 = completely different)
    """
    all_keys = set(list(p.keys()) + list(q.keys()))
    
    epsilon = 1e-10
    p_array = np.array([p.get(k, epsilon) for k in all_keys])
    q_array = np.array([q.get(k, epsilon) for k in all_keys])
    
    p_array = p_array / p_array.sum()
    q_array = q_array / q_array.sum()
    
    return jensenshannon(p_array, q_array)

def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard deviation"""
    return np.mean(values), np.std(values)

def calculate_percentile(values: List[float], percentile: int) -> float:
    """Calculate percentile of values"""
    return np.percentile(values, percentile)

# =============================================================================
# RECOMMENDATION QUALITY METRICS
# =============================================================================

def calculate_ndcg(rankings: List[str], ground_truth: Dict[str, float], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k)
    
    Args:
        rankings: Ordered list of recommended items
        ground_truth: Dict of {item_id: relevance_score}
        k: Number of top items to consider
        
    Returns:
        NDCG score (0-1, higher is better)
    """
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(rankings[:k]):
        relevance = ground_truth.get(item, 0)
        # DCG formula: rel / log2(i+2)
        dcg += relevance / np.log2(i + 2)
    
    # Calculate Ideal DCG (best possible ranking)
    ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    
    # Normalize
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def calculate_precision_at_k(recommendations: List[str], 
                            relevant_items: set, 
                            k: int = 10) -> float:
    """
    Calculate Precision@K
    
    Args:
        recommendations: List of recommended items
        relevant_items: Set of relevant items
        k: Number of top items to consider
        
    Returns:
        Precision score (0-1)
    """
    top_k = set(recommendations[:k])
    relevant_in_top_k = len(top_k.intersection(relevant_items))
    return relevant_in_top_k / k if k > 0 else 0.0

def calculate_recall_at_k(recommendations: List[str],
                          relevant_items: set,
                          k: int = 10) -> float:
    """
    Calculate Recall@K
    
    Args:
        recommendations: List of recommended items
        relevant_items: Set of relevant items
        k: Number of top items to consider
        
    Returns:
        Recall score (0-1)
    """
    top_k = set(recommendations[:k])
    relevant_in_top_k = len(top_k.intersection(relevant_items))
    return relevant_in_top_k / len(relevant_items) if len(relevant_items) > 0 else 0.0

def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_catalog_coverage(all_recommendations: List[List[str]], 
                              catalog_size: int) -> float:
    """
    Calculate catalog coverage (what percentage of items are ever recommended)
    
    Args:
        all_recommendations: List of recommendation lists
        catalog_size: Total number of items in catalog
        
    Returns:
        Coverage percentage (0-100)
    """
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    
    return len(unique_items) / catalog_size * 100 if catalog_size > 0 else 0.0

# =============================================================================
# FAIRNESS METRICS
# =============================================================================

def calculate_demographic_parity(recommendations_by_group: Dict[str, List[List[str]]], 
                                genre_mapping: Optional[Dict[str, List[str]]] = None) -> Tuple[float, Dict]:
    """
    Calculate Demographic Parity
    Measures if different demographic groups receive similar distributions
    
    Args:
        recommendations_by_group: Dict of {group: list_of_recommendations}
        genre_mapping: Optional dict mapping items to genres
        
    Returns:
        (parity_score, detailed_info)
    """
    # Count item/genre distributions for each group
    distributions = {}
    
    for group, recs_list in recommendations_by_group.items():
        item_counts = defaultdict(int)
        
        # Flatten all recommendations for this group
        for recs in recs_list:
            for item in recs:
                item_counts[item] += 1
        
        # Normalize to probability distribution
        total = sum(item_counts.values())
        distributions[group] = {
            item: count / total for item, count in item_counts.items()
        } if total > 0 else {}
    
    # Calculate pairwise divergences between groups
    groups = list(distributions.keys())
    divergences = []
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if distributions[groups[i]] and distributions[groups[j]]:
                div = calculate_js_divergence(
                    distributions[groups[i]], 
                    distributions[groups[j]]
                )
                divergences.append(div)
    
    # Convert to fairness score (lower divergence = higher fairness)
    avg_divergence = np.mean(divergences) if divergences else 0
    parity_score = 1 - avg_divergence  # 0-1 scale, higher is better
    
    return parity_score, {
        'divergences': divergences,
        'distributions': distributions,
        'avg_divergence': avg_divergence
    }

def calculate_individual_fairness(profile_pairs: List[Tuple[str, str]], 
                                  recommendations: Dict[str, List[str]]) -> Tuple[float, List[float]]:
    """
    Calculate Individual Fairness
    Similar users should get similar recommendations
    
    Args:
        profile_pairs: List of (profile_id1, profile_id2) tuples that should be similar
        recommendations: Dict of {profile_id: recommendations_list}
        
    Returns:
        (fairness_score, similarity_scores)
    """
    similarity_scores = []
    
    for pid1, pid2 in profile_pairs:
        if pid1 not in recommendations or pid2 not in recommendations:
            continue
            
        recs1 = set(recommendations[pid1])
        recs2 = set(recommendations[pid2])
        
        # Jaccard similarity
        if len(recs1) == 0 and len(recs2) == 0:
            similarity = 1.0
        else:
            intersection = len(recs1.intersection(recs2))
            union = len(recs1.union(recs2))
            similarity = intersection / union if union > 0 else 0.0
        
        similarity_scores.append(similarity)
    
    # Average similarity across all pairs
    fairness_score = np.mean(similarity_scores) if similarity_scores else 0.0
    
    return fairness_score, similarity_scores

def calculate_equal_opportunity(recommendations_by_group: Dict[str, List[List[str]]], 
                                relevant_items: set) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Equal Opportunity
    Qualified items should have equal chance across groups
    
    Args:
        recommendations_by_group: Dict of {group: list_of_recommendations}
        relevant_items: Set of items considered relevant/qualified
        
    Returns:
        (fairness_score, opportunity_by_group)
    """
    opportunity_scores = {}
    
    for group, recs_list in recommendations_by_group.items():
        # Flatten recommendations
        all_recs = []
        for recs in recs_list:
            all_recs.extend(recs)
        
        # Count how many relevant items were recommended
        recommended_relevant = len(set(all_recs).intersection(relevant_items))
        total_recommended = len(all_recs)
        
        opportunity_scores[group] = (
            recommended_relevant / total_recommended 
            if total_recommended > 0 else 0.0
        )
    
    # Calculate variance (low variance = more fair)
    scores = list(opportunity_scores.values())
    variance = np.var(scores) if scores else 0
    
    # Convert to 0-1 fairness score
    fairness_score = 1 / (1 + variance)
    
    return fairness_score, opportunity_scores

def calculate_exposure_ratio(rankings: List[Tuple[str, str]], 
                            protected_attribute: str = 'gender') -> Tuple[float, Dict[str, float]]:
    """
    Calculate Exposure Ratio
    Measures visibility of different groups in rankings
    
    Args:
        rankings: List of (item_id, group) tuples in ranked order
        protected_attribute: Name of protected attribute
        
    Returns:
        (exposure_ratio, exposure_by_group)
    """
    # Calculate exposure for each position (decreasing with rank)
    exposures_by_group = defaultdict(list)
    
    for position, (item_id, group) in enumerate(rankings):
        # Exposure = 1 / log2(position + 2)
        exposure = 1.0 / np.log2(position + 2)
        exposures_by_group[group].append(exposure)
    
    # Average exposure per group
    avg_exposure = {
        group: np.mean(exposures) 
        for group, exposures in exposures_by_group.items()
    }
    
    # Ratio between highest and lowest
    if not avg_exposure:
        return 1.0, {}
    
    max_exposure = max(avg_exposure.values())
    min_exposure = min(avg_exposure.values())
    
    exposure_ratio = min_exposure / max_exposure if max_exposure > 0 else 1.0
    
    return exposure_ratio, avg_exposure

# =============================================================================
# DATA ORGANIZATION HELPERS
# =============================================================================

def organize_by_attribute(recommendations: Dict[str, Any], 
                          profiles: List[Dict], 
                          attribute: str) -> Dict[str, List]:
    """
    Group recommendations by demographic attribute
    
    Args:
        recommendations: Dict of recommendations
        profiles: List of user profiles
        attribute: Attribute to group by (e.g., 'gender')
        
    Returns:
        Dict of {attribute_value: list_of_recommendations}
    """
    grouped = defaultdict(list)
    
    for profile in profiles:
        attr_value = profile.get(attribute)
        if attr_value is None:
            continue
            
        profile_recs = recommendations.get(profile['id'], [])
        if profile_recs:
            grouped[attr_value].append(profile_recs)
    
    return dict(grouped)

def create_profile_pairs(profiles: List[Dict], 
                        differing_attribute: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Create pairs of profiles that are identical except for one attribute
    
    Args:
        profiles: List of user profiles
        differing_attribute: If specified, only create pairs differing in this attribute
        
    Returns:
        List of (profile_id1, profile_id2) tuples
    """
    pairs = []
    
    for i, p1 in enumerate(profiles):
        for j, p2 in enumerate(profiles[i+1:], start=i+1):
            # Count differences
            differences = []
            for attr in ['gender', 'age', 'occupation']:
                if attr in p1 and attr in p2 and p1[attr] != p2[attr]:
                    differences.append(attr)
            
            # Check if they differ in exactly one attribute
            if len(differences) == 1:
                if differing_attribute is None or differing_attribute in differences:
                    pairs.append((p1['id'], p2['id']))
    
    return pairs

# =============================================================================
# TEXT PROCESSING HELPERS
# =============================================================================

def parse_recommendation_list(text: str) -> List[str]:
    """
    Parse LLM response to extract list of recommendations
    
    Args:
        text: Raw text from LLM
        
    Returns:
        List of recommended item names
    """
    recommendations = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering (1., 2., etc.)
        import re
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        
        # Remove bullet points
        line = re.sub(r'^[\-\*]\s*', '', line)
        
        # Remove quotes
        line = line.strip('"\'')
        
        if line:
            recommendations.append(line)
    
    return recommendations

def extract_movie_title(text: str) -> str:
    """Extract clean movie title from text"""
    import re
    # Remove year in parentheses
    text = re.sub(r'\s*\(\d{4}\)', '', text)
    return text.strip()

# =============================================================================
# RATE LIMITING AND API HELPERS
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit"""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        # Check if we need to wait
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this call
        self.calls.append(time.time())

# =============================================================================
# PROGRESS TRACKING
# =============================================================================

def print_progress(current: int, total: int, prefix: str = ''):
    """Print progress bar"""
    percent = current / total * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # New line when complete

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test some utility functions
    print("Testing utility functions...")
    
    # Test KL divergence
    p = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    q = {'a': 0.4, 'b': 0.4, 'c': 0.2}
    print(f"KL divergence: {calculate_kl_divergence(p, q):.4f}")
    
    # Test NDCG
    rankings = ['item1', 'item2', 'item3']
    ground_truth = {'item1': 5, 'item2': 3, 'item3': 1}
    print(f"NDCG: {calculate_ndcg(rankings, ground_truth):.4f}")
    
    print("\nUtility functions working correctly!")
