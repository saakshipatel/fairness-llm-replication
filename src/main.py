"""

The script runs all three phases of our replication study:
- Phase 1: Bias Detection in ChatGPT
- Phase 2: Cross-Model Evaluation  
- Phase 3: FACTER Mitigation i.e. postprocessing

Usage:
    python src/main.py --all                  # Run all phases
    python src/main.py --phase 1              # Run only Phase 1
    python src/main.py --phase 2              # Run only Phase 2
    python src/main.py --phase 3              # Run only Phase 3
    python src/main.py --quick                # Quick demo mode
"""

import sys
import os
import argparse
import time

# Add src to path
sys.path.append(os.path.dirname(__file__))

import config
from phase1_bias_detection import run_phase1
from phase2_cross_model_eval import run_phase2
from phase3_facter_mitigation import run_phase3

def print_banner():
    banner = """
                                                                        
FAIRNESS IN LLM-BASED RANKING AND RECOMMENDATION SYSTEMS         
A Replication Study                               
                                                                            
  Phase 1: Bias Detection in ChatGPT                                       
  Phase 2: Cross-Model Evaluation                                          
  Phase 3: FACTER Bias Mitigation                                          
                                                                            
    """
    print(banner)

def check_setup():
    """Check if environment is properly set up"""
    print("Checking environment setup...")
    
    issues = []
    
    # Check API keys
    if config.OPENAI_API_KEY == 'your-openai-api-key-here':
        issues.append("⚠️  OpenAI API key not set in config.py or environment")
    else:
        print("  ✓ OpenAI API key configured")
    
    # Check directories
    try:
        config.create_directories()
        print("  ✓ Directories created")
    except Exception as e:
        issues.append(f"⚠️  Could not create directories: {e}")
    
    # Check dataset
    movielens_path = config.DATASETS['movielens']['path']
    if not os.path.exists(movielens_path):
        print(f"  ⚠️  MovieLens dataset not found at {movielens_path}")
        print(f"     Download from: https://grouplens.org/datasets/movielens/1m/")
        print(f"     (Will use synthetic data for demo)")
    else:
        print("  ✓ MovieLens dataset found")
    
    if issues:
        print("\nSetup Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nContinuing anyway...")
    
    print()

def run_all_phases(args):
    """Run all three phases in sequence"""
    results = {}
    
    # Phase 1
    print("\n" + "="*80)
    print("STARTING PHASE 1: BIAS DETECTION")
    print("="*80 + "\n")
    
    start_time = time.time()
    phase1_results = run_phase1(
        model_name=args.model,
        num_profiles=args.profiles_per_combo,
        save_results=True
    )
    results['phase1'] = phase1_results
    print(f"\nPhase 1 completed in {time.time() - start_time:.1f} seconds\n")
    
    # Phase 2
    if not args.skip_phase2:
        print("\n" + "="*80)
        print("STARTING PHASE 2: CROSS-MODEL EVALUATION")
        print("="*80 + "\n")
        
        start_time = time.time()
        phase2_results = run_phase2(
            models=args.models,
            num_items=args.num_items,
            num_pairwise_comparisons=args.num_comparisons,
            save_results=True
        )
        results['phase2'] = phase2_results
        print(f"\nPhase 2 completed in {time.time() - start_time:.1f} seconds\n")
    
    # Phase 3
    print("\n" + "="*80)
    print("STARTING PHASE 3: FACTER MITIGATION")
    print("="*80 + "\n")
    
    start_time = time.time()
    phase3_results = run_phase3(
        phase1_results=phase1_results,
        model_name=args.model,
        num_profiles=args.profiles_per_combo * 3,  # Use subset for efficiency
        save_results=True
    )
    results['phase3'] = phase3_results
    print(f"\nPhase 3 completed in {time.time() - start_time:.1f} seconds\n")
    
    return results

def print_final_summary(results):
    """Print final summary of all phases"""
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL PHASES COMPLETED")
    print("="*80 + "\n")
    
    if 'phase1' in results:
        p1_metrics = results['phase1']['metrics']
        print("Phase 1: Bias Detection")
        print("-" * 80)
        print(f"  Demographic Parity (Gender): {p1_metrics['demographic_parity']['gender']['score']:.4f}")
        print(f"  Individual Fairness:         {p1_metrics['individual_fairness']['score']:.4f}")
        print()
    
    if 'phase2' in results:
        print("Phase 2: Cross-Model Evaluation")
        print("-" * 80)
        for model_name in results['phase2']['model_results'].keys():
            comp = results['phase2']['comparison']['model_fairness'][model_name]
            print(f"  {model_name}: {comp['average_fairness']:.4f}")
        print()
    
    if 'phase3' in results:
        p3_bias = results['phase3']['bias_metrics']
        p3_quality = results['phase3']['quality_metrics']
        print("Phase 3: FACTER Mitigation")
        print("-" * 80)
        print(f"  Bias Reduction Rate:      {p3_bias['bias_reduction_rate']:.2f}%")
        print(f"  Quality Preservation:     {p3_quality['quality_preservation_pct']:.2f}%")
        print()
    
    print("="*80)
    print("\n✓ All phases completed successfully!")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nNext steps:")
    print("  1. Review results in the results/ directory")
    print("  2. Run visualization notebooks in notebooks/")
    print("  3. Analyze findings in your report")
    print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fairness in LLM Replication Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                    # Run all phases
  python main.py --phase 1                # Run only Phase 1
  python main.py --quick                  # Quick demo mode
  python main.py --all --model gpt-4      # Use GPT-4
        """
    )
    
    # Phase selection
    parser.add_argument('--all', action='store_true',
                       help='Run all three phases')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                       help='Run specific phase (1, 2, or 3)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick demo mode (fewer profiles/items)')
    
    # Model settings
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='Primary model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['gpt-3.5-turbo', 'gpt-4'],
                       help='Models for Phase 2 (default: gpt-3.5-turbo gpt-4)')
    
    # Phase 1 settings
    parser.add_argument('--profiles-per-combo', type=int, default=3,
                       help='Profiles per demographic combination (Phase 1)')
    
    # Phase 2 settings
    parser.add_argument('--num-items', type=int, default=20,
                       help='Number of items to rank (Phase 2)')
    parser.add_argument('--num-comparisons', type=int, default=30,
                       help='Number of pairwise comparisons (Phase 2)')
    parser.add_argument('--skip-phase2', action='store_true',
                       help='Skip Phase 2 when running all phases')
    
    # Other options
    parser.add_argument('--no-banner', action='store_true',
                       help='Skip printing banner')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip environment setup check')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.profiles_per_combo = 2
        args.num_items = 10
        args.num_comparisons = 20
        print("Running in QUICK mode (reduced profiles/items for faster execution)\n")
    
    # Print banner
    if not args.no_banner:
        print_banner()
    
    # Check setup
    if not args.skip_check:
        check_setup()
    
    # Run phases
    results = {}
    
    if args.all:
        results = run_all_phases(args)
        print_final_summary(results)
    
    elif args.phase == 1:
        print("\nRunning Phase 1: Bias Detection\n")
        results['phase1'] = run_phase1(
            model_name=args.model,
            num_profiles=args.profiles_per_combo,
            save_results=True
        )
    
    elif args.phase == 2:
        print("\nRunning Phase 2: Cross-Model Evaluation\n")
        results['phase2'] = run_phase2(
            models=args.models,
            num_items=args.num_items,
            num_pairwise_comparisons=args.num_comparisons,
            save_results=True
        )
    
    elif args.phase == 3:
        print("\nRunning Phase 3: FACTER Mitigation\n")
        results['phase3'] = run_phase3(
            model_name=args.model,
            num_profiles=args.profiles_per_combo * 3,
            save_results=True
        )
    
    else:
        parser.print_help()
        print("\nPlease specify --all, --phase, or --quick")
        sys.exit(1)
    
    print("\n✓ Execution completed successfully!")

if __name__ == "__main__":
    main()
