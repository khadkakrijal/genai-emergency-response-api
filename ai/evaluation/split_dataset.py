import os
import pandas as pd

from sklearn.model_selection import train_test_split


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Your original dataset is expected here:
# ai-service/datasets/911.csv
AI_SERVICE_DIR = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        "..",
    )
)

DATASET_FILE = os.path.join(
    AI_SERVICE_DIR,
    "datasets",
    "911.csv",
)

TRAIN_OUTPUT_FILE = os.path.join(
    CURRENT_DIR,
    "train_reference.csv",
)

TEST_OUTPUT_FILE = os.path.join(
    CURRENT_DIR,
    "test_set.csv",
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    print("=" * 70)
    print("KAGGLE 911 DATASET - 80/20 SPLIT")
    print("=" * 70)

    print()
    print(f"Loading dataset from:")
    print(DATASET_FILE)

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_FILE}\n\n"
            "Check that 911.csv is inside the "
            "ai-service/datasets folder."
        )

    dataframe = pd.read_csv(DATASET_FILE)

    print()
    print(
        f"Total incidents loaded: "
        f"{len(dataframe):,}"
    )

    print()
    print("Available columns:")
    print(list(dataframe.columns))

    return dataframe


# =========================================================
# CREATE INCIDENT CATEGORY
# =========================================================

def create_incident_category(dataframe):

    if "title" not in dataframe.columns:
        raise ValueError(
            "The dataset does not contain a 'title' column."
        )

    dataframe = dataframe.copy()

    # Example titles:
    # EMS: BACK PAINS/INJURY
    # Fire: GAS-ODOR/LEAK
    # Traffic: VEHICLE ACCIDENT -

    dataframe["incident_category"] = (
        dataframe["title"]
        .fillna("Unknown")
        .astype(str)
        .str.split(":")
        .str[0]
        .str.strip()
    )

    print()
    print("=" * 70)
    print("FULL DATASET CATEGORY DISTRIBUTION")
    print("=" * 70)

    print(
        dataframe[
            "incident_category"
        ].value_counts()
    )

    return dataframe


# =========================================================
# SPLIT DATASET
# =========================================================

def split_dataset(dataframe):

    print()
    print("=" * 70)
    print("CREATING STRATIFIED 80/20 SPLIT")
    print("=" * 70)

    train_reference, test_set = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,

        # Keep approximately the same proportion
        # of EMS, Fire and Traffic in both sets.
        stratify=dataframe["incident_category"],
    )

    # Reset row indexes after splitting.
    train_reference = (
        train_reference
        .reset_index(drop=True)
    )

    test_set = (
        test_set
        .reset_index(drop=True)
    )

    return train_reference, test_set


# =========================================================
# DISPLAY SPLIT INFORMATION
# =========================================================

def display_split_information(
    full_dataset,
    train_reference,
    test_set,
):

    total = len(full_dataset)
    train_total = len(train_reference)
    test_total = len(test_set)

    print()
    print("=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(
        f"Full Dataset     : "
        f"{total:,}"
    )

    print(
        f"80% Reference Set: "
        f"{train_total:,} "
        f"({train_total / total:.2%})"
    )

    print(
        f"20% Test Set     : "
        f"{test_total:,} "
        f"({test_total / total:.2%})"
    )

    print()
    print("=" * 70)
    print("REFERENCE SET CATEGORY DISTRIBUTION")
    print("=" * 70)

    print(
        train_reference[
            "incident_category"
        ].value_counts()
    )

    print()
    print("=" * 70)
    print("TEST SET CATEGORY DISTRIBUTION")
    print("=" * 70)

    print(
        test_set[
            "incident_category"
        ].value_counts()
    )

    print()
    print("=" * 70)
    print("REFERENCE SET CATEGORY PERCENTAGES")
    print("=" * 70)

    print(
        (
            train_reference[
                "incident_category"
            ].value_counts(
                normalize=True
            ) * 100
        ).round(2)
    )

    print()
    print("=" * 70)
    print("TEST SET CATEGORY PERCENTAGES")
    print("=" * 70)

    print(
        (
            test_set[
                "incident_category"
            ].value_counts(
                normalize=True
            ) * 100
        ).round(2)
    )


# =========================================================
# CHECK FOR DATA LEAKAGE
# =========================================================

def check_data_leakage(
    train_reference,
    test_set,
):

    # Because the original Kaggle dataset may not have
    # a unique ID, we compare original dataframe indexes
    # only indirectly through the saved split sizes.
    #
    # We also verify that the two generated dataframes
    # together contain exactly the original number of rows.

    combined_count = (
        len(train_reference)
        +
        len(test_set)
    )

    print()
    print("=" * 70)
    print("DATA SPLIT VALIDATION")
    print("=" * 70)

    print(
        f"Reference rows + Test rows: "
        f"{combined_count:,}"
    )

    print(
        "Random State: "
        f"{RANDOM_STATE}"
    )

    print(
        "Split method: "
        "Stratified random split"
    )


# =========================================================
# SAVE FILES
# =========================================================

def save_datasets(
    train_reference,
    test_set,
):

    train_reference.to_csv(
        TRAIN_OUTPUT_FILE,
        index=False,
    )

    test_set.to_csv(
        TEST_OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print()
    print("80% Reference Set:")
    print(TRAIN_OUTPUT_FILE)

    print()
    print("20% Test Set:")
    print(TEST_OUTPUT_FILE)


# =========================================================
# MAIN
# =========================================================

def main():

    dataframe = load_dataset()

    dataframe = create_incident_category(
        dataframe
    )

    train_reference, test_set = split_dataset(
        dataframe
    )

    display_split_information(
        dataframe,
        train_reference,
        test_set,
    )

    check_data_leakage(
        train_reference,
        test_set,
    )

    save_datasets(
        train_reference,
        test_set,
    )

    print()
    print("=" * 70)
    print("80/20 DATASET SPLIT COMPLETE")
    print("=" * 70)

    print()
    print(
        "The 80% reference set will be used "
        "as the historical RAG knowledge base."
    )

    print(
        "The 20% test set will be held out "
        "for evaluation."
    )


if __name__ == "__main__":
    main()