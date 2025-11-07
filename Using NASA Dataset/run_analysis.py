

import argparse
import sys
from pathlib import Path
from main import AnomalyDetectionPipeline


def check_data_path(data_path):
    """Verify data path exists"""
    path = Path(data_path)
    if not path.exists():
        print(f"❌ Error: Data path does not exist: {data_path}")
        print("\nPlease download the NASA bearing dataset from:")
        print("https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset")
        print("\nAnd update the data path in this script or provide it as an argument.")
        sys.exit(1)
    return path


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='NASA Bearing Anomaly Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  python run_analysis.py --data-path /path/to/data --output-dir my_results
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='nasa_bearing_dataset',
        help='Path to NASA bearing dataset directory (default: nasa_bearing_dataset)'
    )
    
    parser.add_argument(
        '--bearing',
        type=str,
        default='Bearing1_1',
        help='Bearing name to analyze (default: Bearing1_1)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Number of samples to use (default: 5000, use --full-data for all)'
    )
    
    parser.add_argument(
        '--full-data',
        action='store_true',
        help='Use full dataset (overrides --sample-size)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*70)
    print("NASA BEARING ANOMALY DETECTION")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Data Path:    {args.data_path}")
    print(f"  Bearing:      {args.bearing}")
    print(f"  Sample Size:  {'Full dataset' if args.full_data else args.sample_size}")
    print(f"  Output Dir:   {args.output_dir}")
    print("="*70 + "\n")
    
    # Verify data path
    data_path = check_data_path(args.data_path)
    
    # Determine sample size
    sample_size = None if args.full_data else args.sample_size
    
    # Run pipeline
    try:
        pipeline = AnomalyDetectionPipeline(
            data_path=data_path,
            output_dir=args.output_dir
        )
        
        pipeline.run(
            bearing_name=args.bearing,
            sample_size=sample_size
        )
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nResults saved to: {args.output_dir}/")
        print("\nGenerated files:")
        print("  📊 9 visualization plots")
        print("  📄 1 comprehensive report")
        print("\nNext steps:")
        print("  1. Review visualizations in the output directory")
        print("  2. Read the anomaly_detection_report.txt")
        print("  3. Adjust parameters if needed and re-run")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during analysis: {e}")
        print("\nFor help, run: python run_analysis.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main()