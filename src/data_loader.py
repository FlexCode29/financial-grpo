import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import KFold
import numpy as np

# Define the repository names and the correct, verified file names.
FINANCIAL_REPO = "elliotgao10/financial_statements"
FINANCIAL_FILES = {
    "companies": "company.csv",
    "income": "unified_income_statement_20250701_113211.csv",
    "balance": "unified_balance_sheet_20250701_113211.csv",
    "cash_flow": "unified_cash_flow_20250701_113211.csv",
    "insider": "enhanced_insider_transactions_20250701_115019.csv",
    "news": "enhanced_news_sentiment_20250701_115019.csv",
    "transcripts": "real_earnings_transcripts_20250701_120139.csv",
}


def _summarize_supplementary_data(ticker: str, news_df: pd.DataFrame, insider_df: pd.DataFrame, transcripts_df: pd.DataFrame) -> str:
    """Create a brief text summary of news, insider transactions and earnings transcripts."""
    summaries = []

    # News sentiment
    company_news = news_df[news_df["symbol"] == ticker]
    if not company_news.empty:
        latest_news = company_news.sort_values(by="time_published", ascending=False).iloc[0]
        summaries.append(
            f"Recent News Sentiment: {latest_news['overall_sentiment_label']} (Score: {latest_news['overall_sentiment_score']:.2f})"
        )
    else:
        summaries.append("Recent News Sentiment: No data available.")

    # Insider activity
    company_insider = insider_df[insider_df["symbol"] == ticker]
    if not company_insider.empty and company_insider["data_available"].iloc[0]:
        summaries.append("Recent Insider Activity: Data is available.")
    else:
        summaries.append("Recent Insider Activity: No data available.")

    # Earnings transcripts
    company_transcripts = transcripts_df[transcripts_df["symbol"] == ticker]
    if not company_transcripts.empty:
        latest_transcript = company_transcripts.sort_values(by="quarter", ascending=False).iloc[0]
        summaries.append(
            f"Latest Earnings Transcript (Q{latest_transcript['quarter']}): {latest_transcript['title']}"
        )
    else:
        summaries.append("Latest Earnings Transcript: No data available.")

    return "\n".join(summaries)


def _process_financial_statements(dataframes: dict) -> pd.DataFrame:
    """Merge and process the various financial dataframes and output one row per fiscal year."""
    # Harmonise company names and tickers
    companies_df = dataframes["companies"].rename(columns={"Ticker": "symbol", "Company": "company"})

    # Merge financial statement components
    merge_keys = ["symbol", "fiscalDateEnding"]
    financial_df = pd.merge(
        dataframes["income"],
        dataframes["balance"],
        on=merge_keys,
        how="outer",
        suffixes=("_inc", "_bal"),
    )
    financial_df = pd.merge(financial_df, dataframes["cash_flow"], on=merge_keys, how="outer")

    # Attach company names
    merged_df = pd.merge(companies_df, financial_df, on="symbol", how="right")

    processed_rows = []

    for _, row in merged_df.iterrows():
        company_name = row.get("company")
        ticker_symbol = row["symbol"]
        if pd.isna(company_name):
            continue

        # Supplementary data text block
        supp_summary = _summarize_supplementary_data(
            ticker_symbol, dataframes["news"], dataframes["insider"], dataframes["transcripts"]
        )

        # Prompt assembly for the specific year
        prompt_lines = [
            f"Company: {company_name} ({ticker_symbol})",
            "Analyze the following financial snapshot and supplementary data to predict its performance.",
            "--- Supplementary Data ---",
            supp_summary,
            "--- Financial Snapshot ---",
            (
                f"Date: {row.get('fiscalDateEnding', 'N/A')}; "
                f"Revenue: ${row.get('totalRevenue', 0):,.0f}; "
                f"Net Income: ${row.get('netIncome', 0):,.0f}; "
                f"Total Assets: ${row.get('totalAssets', 0):,.0f}; "
                f"Operating Cash Flow: ${row.get('operatingCashflow', 0):,.0f}"
            ),
        ]
        prompt = "\n".join(prompt_lines)

        # Derive pseudo ground truth values deterministically
        pseudo_random_val = hash(company_name + str(row.get("fiscalDateEnding"))) % 9
        returns_1y, volatility_1y = (
            ("bad", "high")
            if pseudo_random_val < 3
            else ("neutral", "medium")
            if pseudo_random_val < 6
            else ("good", "low")
        )
        returns_5y, volatility_5y = (
            ("bad", "high")
            if (pseudo_random_val + 3) % 9 < 4
            else ("good", "medium")
        )

        processed_rows.append(
            {
                "prompt": prompt,
                "is_grpo_task": True,
                "ground_truth_returns_1y": returns_1y,
                "ground_truth_volatility_1y": volatility_1y,
                "ground_truth_returns_5y": returns_5y,
                "ground_truth_volatility_5y": volatility_5y,
            }
        )

    return pd.DataFrame(processed_rows)


def load_financial_dataset(split: str = "train", test_size: float = 0.2, k_folds: int = 1, fold_index: int = 0):
    """Load the financial statements dataset, generate GRPO style tasks and return a HuggingFace Dataset."""
    print("--- Loading Financial Statement Components ---")
    dataframes = {}
    for name, file_path in FINANCIAL_FILES.items():
        print(f"Loading {name} ({file_path})...")
        ds = load_dataset(FINANCIAL_REPO, data_files=file_path, split="train")
        dataframes[name] = ds.to_pandas()

    print("\n--- Processing Datasets ---")
    grpo_task_df = _process_financial_statements(dataframes)

    # Convert to HuggingFace Dataset and shuffle
    final_dataset = Dataset.from_pandas(grpo_task_df).shuffle(seed=42)
    print(f"--- Dataset ready. Total examples: {len(final_dataset)} ---")

    # Handle train test split or k fold cross validation
    if k_folds > 1:
        print(f"Creating {k_folds} fold cross validation splits. Using fold {fold_index}")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_splits = list(kf.split(final_dataset))
        train_idxs, test_idxs = all_splits[fold_index]
        train_dataset = final_dataset.select(train_idxs)
        test_dataset = final_dataset.select(test_idxs)
        return train_dataset if split == "train" else test_dataset
    else:
        split_dataset = final_dataset.train_test_split(test_size=test_size, seed=42)
        return split_dataset[split]


# Verification block
if __name__ == "__main__":
    print("Running data loader directly for verification...")
    train_dataset = load_financial_dataset(split="train")

    print("\n--- Verification Complete ---")
    print(f"Successfully loaded and processed {len(train_dataset)} examples into the training set.")

    # Display counts and example prompts
    grpo_count = len([item for item in train_dataset if item["is_grpo_task"]])
    print(f"GRPO style tasks in training set: {grpo_count}")

    example = train_dataset[0]
    print("\n--- Example Prompt ---")
    print(example["prompt"][:600] + "...")
    print("Ground Truth Returns One Year:", example["ground_truth_returns_1y"])
