"""
CLI interface for the spend analytics and classification pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from .pipeline import run_pipeline_for_company


def main():
    parser = argparse.ArgumentParser(
        description="Cantill.ai Production Spend Analytics & Zero-Shot Classification Pipeline"
    )
    parser.add_argument("--config", type=str, help="Path to company YAML config file")
    parser.add_argument("--output", type=str, help="Destination CSV path for final submission")
    parser.add_argument("--evaluate", action="store_true", help="Run offline evaluation harness against ground truth labels")
    parser.add_argument("--labels", type=str, help="Path to evaluation labels CSV (optional override)")
    parser.add_argument("--all", action="store_true", help="Run pipeline for both Company A and Company B")

    args = parser.parse_args()

    if args.all:
        print("=== Running Pipeline for Company A ===")
        df_a, m_a = run_pipeline_for_company(
            config_path="configs/company_A.yaml",
            output_path="submission/cantill__A.csv",
            evaluate=True,
            labels_path="company_A/labels_A.csv",
        )

        print("=== Running Pipeline for Company B ===")
        df_b, m_b = run_pipeline_for_company(
            config_path="configs/company_B.yaml",
            output_path="submission/cantill__B.csv",
            evaluate=True,
            labels_path="company_B/seed_labels.csv",
        )

        print("\n================ SUMMARY METRICS ================")
        print("Company A Metrics:")
        print(json.dumps(m_a, indent=2))
        print("\nCompany B Metrics:")
        print(json.dumps(m_b, indent=2))
        return

    if not args.config:
        print("Error: Must provide either --config <path> or --all", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    df, metrics = run_pipeline_for_company(
        config_path=args.config,
        output_path=args.output,
        evaluate=args.evaluate,
        labels_path=args.labels,
    )

    if args.evaluate:
        print("\n================ EVALUATION METRICS ================")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

