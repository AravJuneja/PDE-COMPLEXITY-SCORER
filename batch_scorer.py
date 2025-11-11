import pandas as pd
from pde_scorer import PDEComplexityScorer
import sys
from pathlib import Path


def score_pdes_from_csv(input_file, output_file=None):
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        print(f"Please create a CSV file with columns: name, equation, domain, boundary_conditions")
        return None
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    required_columns = ['name', 'equation', 'domain', 'boundary_conditions']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Required columns: {required_columns}")
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
            print(f"    Complexity Score: {scores['total']}")
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            continue
    
    if not results:
        print("\nNo PDEs were successfully scored.")
        return None
    
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
        print(f"\nResults saved to: {output_file}")
    
    return results_df


def score_pdes_with_predictions(input_file, output_file=None):
    
    results_df = score_pdes_from_csv(input_file, output_file=None)
    
    if results_df is None:
        return None
    
    results_df['predicted_l2_error'] = -6.11 + 2.73 * results_df['total']
    results_df['predicted_l2_error'] = results_df['predicted_l2_error'].round(2)
    
    print("\n" + "="*70)
    print("PERFORMANCE PREDICTIONS")
    print("="*70)
    print("Model: L2 Error (%) = -6.11 + 2.73 × Complexity Score")
    print("Correlation: r = 0.845 (based on 8 PDEs from QCPINN paper + extensions)")
    print("\nInterpretation:")
    print("  Complexity 0-4:  Use classical PINN (quantum overkill)")
    print("  Complexity 5-9:  Use QCPINN (sweet spot for quantum advantage)")
    print("  Complexity 10+:  Both methods struggle (research frontier)")
    print()
    
    prediction_df = results_df[['name', 'total', 'predicted_l2_error']].copy()
    prediction_df.columns = ['PDE', 'Complexity', 'Predicted L2 Error (%)']
    print(prediction_df.to_string(index=False))
    
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return results_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Score PDE complexity from CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example CSV format (save as 'pdes.csv'):

name,equation,domain,boundary_conditions,notes
Helmholtz,laplacian(u) + k**2 * u = f,"[-1,1] x [-1,1]",u = h on boundary,Test case
Burgers,u_t + u*u_x = nu*u_xx,"[0,1] x [0,T]","u(0,t)=0, u(1,t)=0",Nonlinear

Usage examples:
  python batch_scorer.py pdes.csv
  python batch_scorer.py pdes.csv -o results.csv
  python batch_scorer.py pdes.csv -o results.csv --predict
        """
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
        help='Include L2 error predictions based on complexity'
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
        print("\nScores PDEs from CSV file using complexity framework")
        print("Based on: Farea et al. (2025) QCPINN paper analysis\n")
        print("Usage:")
        print("  python batch_scorer.py input.csv")
        print("  python batch_scorer.py input.csv -o results.csv")
        print("  python batch_scorer.py input.csv -o results.csv --predict\n")
        print("Example:")
        print("  python batch_scorer.py data/qcpinn_pdes.csv -o data/results.csv -p\n")
        print("For detailed help:")
        print("  python batch_scorer.py --help")
    else:
        main()