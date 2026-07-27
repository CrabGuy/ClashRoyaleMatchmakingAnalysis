from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_matches(file_name, hf_repo=None):
    local_path = Path("data/processed") / file_name
    if local_path.exists():
        return pd.read_parquet(local_path)
    if hf_repo:
        return pd.read_parquet(f"hf://datasets/{hf_repo}/{file_name}")
    raise FileNotFoundError(
        f"Could not find {local_path} and no hf_repo provided."
    )


def assign_strata(df, trophy_bin_size=500):
    trophy_bracket = (
        df["match_average_trophies"] // trophy_bin_size * trophy_bin_size
    ).astype(int)
    time_bucket = (
        pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%S.%fZ")
        .dt.to_period("W")
        .astype(str)
    )
    return df.assign(stratum=trophy_bracket.astype(str) + "_" + time_bucket)


def get_exploded_pairs(df):
    df = df.assign(
        p1=df["player1_deck"].str.split("|"), p2=df["player2_deck"].str.split("|")
    )
    p1_exp = df.explode("p1")[["stratum", "p1"]]
    p1_exp["match_id"] = p1_exp.index
    p2_exp = df.explode("p2")[["p2"]]
    p2_exp["match_id"] = p2_exp.index

    return pd.merge(p1_exp, p2_exp, on="match_id")


def compute_deviations(merged_pairs, min_matches=200):
    crosstab = pd.crosstab(
        [merged_pairs["stratum"], merged_pairs["p1"]], merged_pairs["p2"]
    )

    stratum_totals = crosstab.groupby(level=0).sum()
    stratum_probs = stratum_totals.div(stratum_totals.sum(axis=1), axis=0)

    row_totals = crosstab.sum(axis=1)
    valid_rows = row_totals >= min_matches
    crosstab_valid = crosstab[valid_rows]
    row_totals_valid = row_totals[valid_rows]

    cond_probs = crosstab_valid.div(row_totals_valid, axis=0)
    baseline_aligned = stratum_probs.reindex(cond_probs.index, level=0)

    abs_deviations = (cond_probs - baseline_aligned).abs().mean(axis=1)

    weights = row_totals_valid
    weighted_dev = abs_deviations * weights

    return weighted_dev.groupby(level=1).sum() / weights.groupby(level=1).sum()


def plot_all_deviations(agg_dev):
    plot_data = agg_dev.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(
        plot_data.index, plot_data.values, color="steelblue", edgecolor="black"
    )

    ax.set_ylabel("Mean Absolute Deviation")
    ax.set_title(
        "Average Distribution Shift by Card (Weighted Across All Strata)"
    )
    plt.xticks(rotation=90, fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    THRESHOLD_PERCENTAGE = 1.5
    FILE_NAME = "trophy_battles.parquet"
    HF_REPO = "your-username/your-dataset-repo"

    df = load_matches(FILE_NAME, hf_repo=HF_REPO)
    df = assign_strata(df)
    pairs = get_exploded_pairs(df)
    deviations = compute_deviations(pairs)

    overall_mean = deviations.mean()
    print(
        f"Overall mean absolute deviation across all cards: {overall_mean:.4%}"
    )

    if overall_mean * 100 <= THRESHOLD_PERCENTAGE:
        print(
            "This shows there is no significant deviation from the baseline distribution."
        )
        print(
            "Conclusion: The matchmaking algorithm appears random and does not pair you based on card choices."
        )
        plot_all_deviations(deviations)
    else:
        print(
            "Conclusion: The overall mean deviation exceeds the noise threshold."
        )
        print(
            "Further investigation is needed to determine if specific cards drive this shift."
        )


if __name__ == "__main__":
    main()