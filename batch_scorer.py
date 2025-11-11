import pandas as pd
from pde_scorer import PDEComplexityScorer
import sys
from pathlib import Path


def score_pdes_from_csv(input_file, output_file=None):
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        return None
    
    df = pd.read_csv(input_file)
    
    required_columns = ['name', 'equation', 'domain', 'boundary_conditions']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return None
    
    results = []
    
    print("="*70)
    print("SCORING PDEs FROM FILE")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Total PDEs: {len(df)}\n")
    
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] Scoring: {row['name']}")
        
        try:
            scorer = PDEComplexityScorer(
                equation=row['equation'],
                domain=row['domain'],
                boundary_conditions=row['boundary_conditions']
            )
            
            scores = scorer.get_total_score()
            scores['name'] = row['name']
            
            if 'notes' in row and pd.notna(row['notes']):
                scores['notes'] = row['notes']
            
            results.append(scores)
            print(f"    Score: {scores['total']}")
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            continue
    
    results_df = pd.DataFrame(results)
    
    column_order = ['name', 'total', 'dimensionality', 'nonlinearity', 
                    'boundary', 'time', 'coupling']
    if 'notes' in results_df.columns:
        column_order.append('notes')
    
    results_df = results_df[column_order]
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    return results_df


def score_pdes_with_predictions(input_file, output_file=None):
    
    results_df = score_pdes_from_csv(input_file, output_file=None)
    
    if results_df is None:
        return
    
    results_df['predicted_l2_error'] = -6.11 + 2.73 * results_df['total']
    results_df['predicted_l2_error'] = results_df['predicted_l2_error'].round(2)
    
    print("\n" + "="*70)
    print("PERFORMANCE PREDICTIONS")
    print("="*70)
    print("Using model: L2 Error (%) = -6.11 + 2.73 × Complexity Score")
    print("Correlation: r = 0.845 (based on 8 PDEs)\n")
    
    prediction_df = results_df[['name', 'total', 'predicted_l2_error']].copy()
    prediction_df.columns = ['PDE', 'Complexity', 'Predicted L2 Error (%)']
    print(prediction_df.to_string(index=False))
    
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    return results_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Score PDE complexity from CSV file'
    )
    parser.add_argument(
        'input_file',
        help='Path to input CSV file with PDEs'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to output CSV file for results',
        default=None
    )
    parser.add_argument(
        '-p', '--predict',
        action='store_true',
        help='Include L2 error predictions'
    )
    
    args = parser.parse_args()
    
    if args.predict:
        score_pdes_with_predictions(args.input_file, args.output)
    else:
        score_pdes_from_csv(args.input_file, args.output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("="*70)
        print("BATCH PDE COMPLEXITY SCORER")
        print("="*70)
        print("\nUsage:")
        print("  python batch_scorer.py input.csv")
        print("  python batch_scorer.py input.csv -o results.csv")
        print("  python batch_scorer.py input.csv -o results.csv --predict")
        print("\nExample:")
        print("  python batch_scorer.py data/qcpinn_pdes.csv -o data/results.csv -p")
        print("\nFor help:")
        print("  python batch_scorer.py --help")
    else:
        main()