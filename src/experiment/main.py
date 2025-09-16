#!/usr/bin/env python3
"""
Main entry point for running the LLM monetary priming experiment.

This script provides a command-line interface to run different types of experiments
using the ExperimentRunner class.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from experiment.runner import ExperimentRunner
from analysis.trends_analysis import analyze_all


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM monetary priming experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run open source models experiment
  python main.py --experiment open_source --hf-token YOUR_TOKEN --output results_open_source.csv
  
  # Run Gemini experiment
  python main.py --experiment gemini --gemini-key YOUR_KEY --output results_gemini.csv
  
  # Run both experiments
  python main.py --experiment both --hf-token YOUR_TOKEN --gemini-key YOUR_KEY --output all_results.csv
  
  # Run with custom model selection
  python main.py --experiment open_source --hf-token YOUR_TOKEN --models Qwen/Qwen3-4B --output custom_results.csv
        """
    )
    
    # Experiment type
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        choices=["open_source", "gemini", "both"],
        default="both",
        help="Type of experiment to run (default: both)"
    )
    
    # API keys
    parser.add_argument(
        "--hf-token", "-t",
        type=str,
        help="HuggingFace API token for open source models"
    )
    
    parser.add_argument(
        "--gemini-key", "-g",
        type=str,
        help="Google Gemini API key"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="experiment_results.csv",
        help="Output CSV filename (default: experiment_results.csv)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    # Model selection (for open source experiments)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (overrides config.py settings)"
    )
    
    # Experiment options
    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="Clear previous results before running new experiment"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually executing"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check required API keys based on experiment type
    if args.experiment in ["open_source", "both"] and not args.hf_token:
        errors.append("--hf-token is required for open source experiments")
    
    if args.experiment in ["gemini", "both"] and not args.gemini_key:
        errors.append("--gemini-key is required for Gemini experiments")
    
    # Check if output directory exists or can be created
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory '{args.output_dir}': {e}")
    
    if errors:
        print("Error: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def setup_environment(args):
    """Setup environment variables and paths."""
    # Set environment variables for API keys
    if args.hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = args.hf_token
    
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def run_experiment(args, output_dir):
    """Run the specified experiment(s)."""
    print(f"Starting experiment: {args.experiment}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    # Initialize runner
    runner = ExperimentRunner(
        huggingface_token=args.hf_token,
        gemini_api_key=args.gemini_key
    )
    
    # Setup authentication
    if args.experiment in ["open_source", "both"]:
        print("Setting up HuggingFace authentication...")
        runner.setup()
    
    # Clear results if requested
    if args.clear_results:
        print("Clearing previous results...")
        runner.clear_results()
    
    results_df = None
    
    try:
        # Run experiments based on type
        if args.experiment == "open_source":
            print("Running open source models experiment...")
            results_df = runner.run_open_source_experiment()
            
        elif args.experiment == "gemini":
            print("Running Gemini experiment...")
            results_df = runner.run_gemini_experiment()
            
        elif args.experiment == "both":
            print("Running both experiments...")
            
            # Run open source first
            print("\n1. Running open source models experiment...")
            open_source_results = runner.run_open_source_experiment()
            
            # Clear results for Gemini
            runner.clear_results()
            
            # Run Gemini
            print("\n2. Running Gemini experiment...")
            gemini_results = runner.run_gemini_experiment()
            
            # Combine results
            results_df = runner.get_results_dataframe()
        
        # Save results
        if results_df is not None and not results_df.empty:
            output_path = output_dir / args.output
            runner.save_results(str(output_path))
            
            print(f"\nExperiment completed successfully!")
            print(f"Results saved to: {output_path}")
            
            # Display results summary (like in the notebook)
            print("\n" + "=" * 50)
            print("EXPERIMENT RESULTS SUMMARY")
            print("=" * 50)
            
            # Show a clean summary of results
            if all(col in results_df.columns for col in ['model', 'prefix_type', 'paraphrase_index', 'response']):
                summary_df = results_df[['model', 'prefix_type', 'paraphrase_index', 'response']].copy()
                print("\nSummary of Results:")
                print(summary_df.to_string(index=False))
            
            # Show comprehensive statistics
            print(f"\nTotal experiments run: {len(results_df)}")
            if "model" in results_df.columns:
                print(f"Models tested: {results_df['model'].nunique()}")
                model_counts = results_df["model"].value_counts()
                print("\nModel breakdown:")
                for model, count in model_counts.items():
                    print(f"  {model}: {count} experiments")
            
            if "prefix_type" in results_df.columns:
                print(f"Prefix conditions: {results_df['prefix_type'].nunique()}")
                prefix_counts = results_df["prefix_type"].value_counts()
                print("\nPrefix type breakdown:")
                for prefix, count in prefix_counts.items():
                    print(f"  {prefix}: {count} experiments")
                    
            if "paraphrase_index" in results_df.columns:
                print(f"Paraphrases tested: {results_df['paraphrase_index'].nunique()}")
                if "prefix_type" in results_df.columns:
                    total_combinations = results_df['prefix_type'].nunique() * results_df['paraphrase_index'].nunique()
                    print(f"Total combinations per model: {results_df['prefix_type'].nunique()} prefixes × {results_df['paraphrase_index'].nunique()} paraphrases = {total_combinations}")
            
            # Analyze decision token probabilities and top tokens (for open source models only)
            print("\n" + "=" * 50)
            print("DECISION TOKEN PROBABILITY ANALYSIS")
            print("=" * 50)
            
            token_analysis_count = 0
            for idx, row in results_df.iterrows():
                if row.get('decision_tokens') is not None:
                    token_analysis_count += 1
                    print(f"\nModel: {row['model']}")
                    print(f"Prefix: {row['prefix_type']}")
                    print(f"Paraphrase: {row['paraphrase_index']}")
                    print(f"Response: {row['response']}")
                    print("Decision token probabilities:")
                    
                    # Handle different formats of decision_tokens
                    if isinstance(row['decision_tokens'], list):
                        for token, prob in row['decision_tokens']:
                            print(f"  {token}: {prob:.15f}")
                    elif isinstance(row['decision_tokens'], dict):
                        for token, prob in row['decision_tokens'].items():
                            print(f"  {token}: {prob:.15f}")
                    
                    # Print top k tokens if available
                    if 'top_tokens' in row and row['top_tokens'] is not None:
                        print("Top k most probable next tokens:")
                        if isinstance(row['top_tokens'], list):
                            for i, (token, prob) in enumerate(row['top_tokens'], 1):
                                print(f"  {i}. '{token}': {prob:.15f}")
                        elif isinstance(row['top_tokens'], dict):
                            for i, (token, prob) in enumerate(row['top_tokens'].items(), 1):
                                print(f"  {i}. '{token}': {prob:.15f}")
                    
                    print("-" * 30)
            
            if token_analysis_count == 0:
                print("No decision token probability data available (this is normal for closed-source models like Gemini)")
            
            # Additional analysis: Response patterns and insights
            print("\n" + "=" * 50)
            print("RESPONSE PATTERN ANALYSIS")
            print("=" * 50)
            
            # Analyze response patterns by prefix type
            if "prefix_type" in results_df.columns and "response" in results_df.columns:
                print("\nResponse patterns by prefix type:")
                for prefix in results_df['prefix_type'].unique():
                    prefix_responses = results_df[results_df['prefix_type'] == prefix]['response']
                    print(f"\nPrefix: {prefix}")
                    print(f"  Number of responses: {len(prefix_responses)}")
                    print(f"  Unique responses: {prefix_responses.nunique()}")
                    
                    # Show sample responses
                    sample_responses = prefix_responses.head(3).tolist()
                    print(f"  Sample responses:")
                    for i, resp in enumerate(sample_responses, 1):
                        print(f"    {i}. {resp}")
            
            # Analyze by model
            if "model" in results_df.columns and "response" in results_df.columns:
                print("\nResponse patterns by model:")
                for model in results_df['model'].unique():
                    model_responses = results_df[results_df['model'] == model]['response']
                    print(f"\nModel: {model}")
                    print(f"  Number of responses: {len(model_responses)}")
                    print(f"  Unique responses: {model_responses.nunique()}")
                    
                    # Show sample responses
                    sample_responses = model_responses.head(3).tolist()
                    print(f"  Sample responses:")
                    for i, resp in enumerate(sample_responses, 1):
                        print(f"    {i}. {resp}")
            
            # Summary statistics
            print("\n" + "=" * 50)
            print("FINAL SUMMARY")
            print("=" * 50)
            print(f"Results saved to: {output_path}")
            print(f"Total experiments run: {len(results_df)}")
            print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
                    
        else:
            print("No results generated from experiment.")
            
    except Exception as e:
        print(f"Error running experiment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False
    
    return True


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Setup environment
    try:
        output_dir = setup_environment(args)
    except Exception as e:
        print(f"Error setting up environment: {e}")
        sys.exit(1)
    
    # Show dry run info if requested
    if args.dry_run:
        print("DRY RUN - No experiments will be executed")
        print(f"Experiment type: {args.experiment}")
        print(f"Output: {output_dir / args.output}")
        if args.models:
            print(f"Models: {', '.join(args.models)}")
        print("Use --dry-run=false to actually run the experiment")
        return
    
    # Run experiment
    success = run_experiment(args, output_dir)
    analyze_all()
    
    if success:
        print("\nExperiment completed successfully!")
        sys.exit(0)
    else:
        print("\nExperiment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
