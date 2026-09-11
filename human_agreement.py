import pandas as pd
from sklearn.metrics import cohen_kappa_score

FILE = "amazonhelp_response_eval_30_to_rate.csv"


def main():
    df = pd.read_csv(FILE)

    print("Columns available:")
    print(df.columns.tolist())

    if "human_quality" not in df.columns:
        raise ValueError(
            "human_quality column not found."
        )

    if "second_human_quality" not in df.columns:
        df["second_human_quality"] = pd.NA

    rated = df.dropna(
        subset=["human_quality", "second_human_quality"]
    )

    if len(rated) == 0:
        print(
            "\nNo second-rater scores yet.\n"
            "Fill both human_quality and "
            "second_human_quality to calculate agreement."
        )
        return

    kappa = cohen_kappa_score(
        rated["human_quality"],
        rated["second_human_quality"]
    )

    print(f"\nRated examples: {len(rated)}")
    print(f"Cohen's kappa: {kappa:.3f}")


if __name__ == "__main__":
    main()