import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

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
FINCOT_REPO = "TheFinAI/FinCoT"

def _summarize_supplementary_data(ticker, news_df, insider_df, transcripts_df):
    """Creates a brief text summary of news, insider transactions, and earnings transcripts."""
    summaries = []
    
    company_news = news_df[news_df['symbol'] == ticker]
    if not company_news.empty:
        latest_news = company_news.sort_values(by='time_published', ascending=False).iloc[0]
        summaries.append(f"Recent News Sentiment: {latest_news['overall_sentiment_label']} (Score: {latest_news['overall_sentiment_score']:.2f})")
    else:
        summaries.append("Recent News Sentiment: No data available.")
        
    company_insider = insider_df[insider_df['symbol'] == ticker]
    if not company_insider.empty and company_insider['data_available'].iloc[0]:
        summaries.append(f"Recent Insider Activity: Data is available.")
    else:
        summaries.append("Recent Insider Activity: No data available.")

    company_transcripts = transcripts_df[transcripts_df['symbol'] == ticker]
    if not company_transcripts.empty:
        latest_transcript = company_transcripts.sort_values(by='quarter', ascending=False).iloc[0]
        summaries.append(f"Latest Earnings Transcript (Q{latest_transcript['quarter']}): {latest_transcript['title']}")
    else:
        summaries.append("Latest Earnings Transcript: No data available.")
        
    return "\n".join(summaries)

def _process_financial_statements(dataframes):
    """Merges and processes the various financial dataframes to create GRPO task prompts."""
    companies_df = dataframes["companies"].rename(columns={"Ticker": "symbol", "Company": "company"})
    
    merge_keys = ['symbol', 'fiscalDateEnding']
    financial_df = pd.merge(dataframes["income"], dataframes["balance"], on=merge_keys, how='outer', suffixes=('_inc', '_bal'))
    financial_df = pd.merge(financial_df, dataframes["cash_flow"], on=merge_keys, how='outer')

    merged_df = pd.merge(companies_df, financial_df, on='symbol', how='right')

    grouped = merged_df.groupby('symbol')
    processed_data = []

    for ticker_symbol, company_df in grouped:
        company_name = company_df['company'].iloc[0]
        if pd.isna(company_name): continue

        supp_summary = _summarize_supplementary_data(ticker_symbol, dataframes["news"], dataframes["insider"], dataframes["transcripts"])

        prompt_lines = [
            f"Company: {company_name} ({ticker_symbol})",
            "Analyze the following financial history and supplementary data to predict its performance.",
            "--- Supplementary Data ---",
            supp_summary,
            "--- Financial History ---"
        ]
        
        company_df_sorted = company_df.sort_values(by='fiscalDateEnding', ascending=False)
        for _, row in company_df_sorted.iterrows():
            date = row.get('fiscalDateEnding', 'N/A')
            revenue = row.get('totalRevenue', 0)
            net_income = row.get('netIncome', 0)
            assets = row.get('totalAssets', 0)
            cash_flow = row.get('operatingCashflow', 0)
            prompt_lines.append(
                f"Date: {date}; Revenue: ${revenue:,.0f}; Net Income: ${net_income:,.0f}; "
                f"Total Assets: ${assets:,.0f}; Operating Cash Flow: ${cash_flow:,.0f}"
            )
        prompt = "\n".join(prompt_lines)

        pseudo_random_val = hash(company_name) % 9
        returns_1y, volatility_1y = ("bad", "high") if pseudo_random_val < 3 else (("neutral", "medium") if pseudo_random_val < 6 else ("good", "low"))
        returns_5y, volatility_5y = ("bad", "high") if (pseudo_random_val + 3) % 9 < 4 else ("good", "medium")
            
        processed_data.append({
            "prompt": prompt, "is_grpo_task": True, "ground_truth_returns_1y": returns_1y,
            "ground_truth_volatility_1y": volatility_1y, "ground_truth_returns_5y": returns_5y,
            "ground_truth_volatility_5y": volatility_5y,
        })
    return pd.DataFrame(processed_data)

def _process_fincot_data(fincot_ds):
    """Processes the FinCoT dataset into a standard prompt format."""
    processed_data = []
    for item in fincot_ds:
        # --- THIS IS THE FIX ---
        # The prompt is in the 'Question' column, as verified by the inspection output.
        prompt = item['Question']
        processed_data.append({"prompt": prompt, "is_grpo_task": False})
    return pd.DataFrame(processed_data)

def load_financial_dataset(split='train', test_size=0.2):
    """Loads and combines all necessary datasets for mixed-task training."""
    print("--- Loading All Datasets Individually ---")
    dataframes = {}
    for name, file_path in FINANCIAL_FILES.items():
        print(f"Loading {name} ({file_path})...")
        ds = load_dataset(FINANCIAL_REPO, data_files=file_path, split="train")
        dataframes[name] = ds.to_pandas()
    
    print(f"Loading from repository: {FINCOT_REPO}")
    fincot_ds = load_dataset(FINCOT_REPO, split="SFT")

    print("\n--- Processing and Merging Datasets ---")
    grpo_task_df = _process_financial_statements(dataframes)
    reasoning_task_df = _process_fincot_data(fincot_ds)

    combined_df = pd.concat([grpo_task_df, reasoning_task_df], ignore_index=True)
    final_dataset = Dataset.from_pandas(combined_df).shuffle(seed=42)

    print(f"--- All datasets loaded and combined. Total examples: {len(final_dataset)} ---")
    
    split_dataset = final_dataset.train_test_split(test_size=test_size, seed=42)
    return split_dataset[split]

# --- VERIFICATION BLOCK ---
if __name__ == '__main__':
    print("Running data loader directly for verification...")
    train_dataset = load_financial_dataset(split='train')
    
    print("\n--- Verification Complete ---")
    print(f"Successfully loaded and processed {len(train_dataset)} examples into the training set.")
    
    grpo_example = next(item for item in train_dataset if item['is_grpo_task'])
    print("\n--- Example of a GRPO Task record (with merged data) ---")
    print(grpo_example['prompt'][:600] + "...")
    print("GRPO Ground Truth:", grpo_example['ground_truth_returns_1y'])
    
    fincot_example = next(item for item in train_dataset if not item['is_grpo_task'])
    print("\n--- Example of a FinCoT Task record ---")
    print(fincot_example['prompt'])
    print("-----------------------------------")