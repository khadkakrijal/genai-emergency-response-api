import os
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTIONS_FILE = os.path.join(
    CURRENT_DIR,
    "holdout_predictions.csv"
)

ERROR_OUTPUT_FILE = os.path.join(
    CURRENT_DIR,
    "holdout_errors.csv"
)


def main():

    print("=" * 70)
    print("HOLDOUT ERROR ANALYSIS")
    print("=" * 70)

    # Load previous evaluation results
    df = pd.read_csv(PREDICTIONS_FILE)

    print(f"\nTotal predictions: {len(df)}")

    # Select incorrect predictions
    errors = df[df["correct"] == False].copy()

    print(f"Correct predictions: {len(df) - len(errors)}")
    print(f"Incorrect predictions: {len(errors)}")

    # -----------------------------------------------------
    # Error types
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MISCLASSIFICATION TYPES")
    print("=" * 70)

    error_summary = (
        errors.groupby(
            ["expected_category", "predicted_category"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(error_summary.to_string(index=False))

    # -----------------------------------------------------
    # Traffic errors
    # -----------------------------------------------------

    traffic_errors = errors[
        errors["expected_category"] == "Traffic"
    ]

    print("\n" + "=" * 70)
    print("TRAFFIC MISCLASSIFICATIONS")
    print("=" * 70)

    print(f"\nTraffic errors: {len(traffic_errors)}")

    for _, row in traffic_errors.iterrows():

        print("\n" + "-" * 70)

        print(
            f"Original title : {row['original_title']}"
        )

        print(
            f"Expected       : {row['expected_category']}"
        )

        print(
            f"Predicted      : {row['predicted_category']}"
        )

        print(
            f"Similarity     : {row['highest_similarity']}"
        )

        print(
            f"Precision@5    : {row['precision_at_5']}"
        )

        print(
            f"Query          : {row['query']}"
        )

        print(
            f"Top categories : {row['top_5_categories']}"
        )

        print(
            f"Top titles     : {row['top_5_titles']}"
        )

    # -----------------------------------------------------
    # Similarity statistics
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("SIMILARITY ANALYSIS")
    print("=" * 70)

    correct_rows = df[df["correct"] == True]

    print(
        f"\nMean similarity - Correct: "
        f"{correct_rows['highest_similarity'].mean():.4f}"
    )

    print(
        f"Mean similarity - Incorrect: "
        f"{errors['highest_similarity'].mean():.4f}"
    )

    print(
        f"Minimum similarity - Correct: "
        f"{correct_rows['highest_similarity'].min():.4f}"
    )

    print(
        f"Maximum similarity - Incorrect: "
        f"{errors['highest_similarity'].max():.4f}"
    )

    # -----------------------------------------------------
    # Save errors
    # -----------------------------------------------------

    errors.to_csv(
        ERROR_OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nErrors saved to:")
    print(ERROR_OUTPUT_FILE)


if __name__ == "__main__":
    main()