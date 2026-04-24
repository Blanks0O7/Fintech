"""
Phase 4 Data Pipeline: 100 Random S&P 500 Stocks with 10-K Filings
=================================================================
1. Scrape S&P 500 tickers from Wikipedia
2. Randomly sample 100 (seeded)
3. Download daily close prices 2015-2024 via yfinance
4. Extract 10-K Item 1 (Business Description) from SEC EDGAR
5. Compute 100x100 TF-IDF cosine similarity matrix
6. Map GICS sectors via yfinance
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import os
import re
import time
import random
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── SEC EDGAR headers (required by SEC) ─────────────────────────────
EDGAR_HEADERS = {
    "User-Agent": "ResearchProject radheshyam@university.edu",
    "Accept-Encoding": "gzip, deflate",
}


def get_sp500_tickers():
    """Scrape current S&P 500 tickers from Wikipedia."""
    print("Scraping S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    sectors = dict(zip(
        [t.replace(".", "-") for t in df["Symbol"]],
        df["GICS Sector"]
    ))
    print(f"  Found {len(tickers)} tickers")
    return tickers, sectors


def sample_stocks(tickers, sectors, n=100):
    """Randomly sample n tickers, ensuring we get diverse sectors."""
    random.shuffle(tickers)
    selected = tickers[:n]
    selected.sort()
    sel_sectors = {t: sectors.get(t, "Unknown") for t in selected}
    print(f"\nSelected {len(selected)} tickers across {len(set(sel_sectors.values()))} GICS sectors:")
    from collections import Counter
    for sector, count in Counter(sel_sectors.values()).most_common():
        print(f"  {sector}: {count}")
    return selected, sel_sectors


def download_prices(tickers, start="2015-01-01", end="2024-01-01"):
    """Download daily close prices."""
    print(f"\nDownloading prices for {len(tickers)} tickers ({start} to {end})...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    # Drop tickers with >30% missing data
    missing = df.isnull().mean()
    drop = missing[missing > 0.3].index.tolist()
    if drop:
        print(f"  Dropping {len(drop)} tickers with >30% missing: {drop}")
        df = df.drop(columns=drop)
    df = df.ffill().bfill().dropna(axis=1)
    print(f"  Final: {len(df.columns)} tickers, {len(df)} days")
    return df


def extract_item1(html_text):
    """
    Extract Item 1 (Business Description) from a 10-K filing HTML.
    Looks for text between 'Item 1.' and 'Item 1A.' or 'Item 2.'
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    # Try to find Item 1 boundaries
    patterns = [
        (r"(?i)item\s*1[\.\\s]+business", r"(?i)item\s*1a[\.\\s]"),
        (r"(?i)item\s*1[\.\\s]+business", r"(?i)item\s*2[\.\\s]"),
        (r"(?i)ITEM\s+1[\.\\s]+BUSINESS", r"(?i)ITEM\s+1A"),
    ]
    
    for start_pat, end_pat in patterns:
        start_match = re.search(start_pat, text)
        if start_match:
            end_match = re.search(end_pat, text[start_match.end():])
            if end_match:
                extracted = text[start_match.end():start_match.end() + end_match.start()]
                # Clean up
                extracted = re.sub(r"\s+", " ", extracted).strip()
                if len(extracted) > 200:
                    # Cap at ~10K words to avoid memory issues
                    words = extracted.split()[:10000]
                    return " ".join(words)
    
    return None


def get_10k_text_edgar_v2(ticker, max_retries=2):
    """
    Use EDGAR company filings API to get 10-K text.
    """
    try:
        # Get CIK from ticker
        cik_resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=15
        )
        if cik_resp.status_code != 200:
            return None
        
        cik_data = cik_resp.json()
        cik = None
        for entry in cik_data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break
        
        if not cik:
            return None
        
        # Get filings index
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        filings_resp = requests.get(filings_url, headers=EDGAR_HEADERS, timeout=15)
        if filings_resp.status_code != 200:
            return None
        
        filings_data = filings_resp.json()
        recent = filings_data.get("filings", {}).get("recent", {})
        
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        
        # Find most recent 10-K
        for i, form in enumerate(forms):
            if form == "10-K":
                accession = accessions[i].replace("-", "")
                doc = primary_docs[i]
                
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{doc}"
                doc_resp = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=30)
                
                if doc_resp.status_code == 200:
                    text = doc_resp.text
                    item1 = extract_item1(text)
                    if item1 and len(item1) > 200:
                        return item1
                break
        
        time.sleep(0.15)  # SEC rate limit: 10 req/sec
        
    except Exception as e:
        pass
    
    return None


def get_business_descriptions(tickers):
    """
    Get business descriptions for all tickers.
    Strategy: Try SEC EDGAR 10-K first, fall back to Yahoo Finance.
    """
    print(f"\nFetching business descriptions for {len(tickers)} tickers...")
    descriptions = {}
    sources = {}
    
    # First pass: Try EDGAR 10-K
    print("  Pass 1: Trying SEC EDGAR 10-K filings...")
    for i, ticker in enumerate(tickers):
        print(f"    [{i+1}/{len(tickers)}] {ticker}...", end=" ")
        
        text = get_10k_text_edgar_v2(ticker)
        if text:
            descriptions[ticker] = text
            sources[ticker] = "10-K"
            print(f"10-K ({len(text.split())} words)")
        else:
            print("no 10-K found")
        
        time.sleep(0.15)  # SEC rate limit
    
    # Second pass: Yahoo Finance fallback
    missing = [t for t in tickers if t not in descriptions]
    if missing:
        print(f"\n  Pass 2: Yahoo Finance fallback for {len(missing)} tickers...")
        for ticker in missing:
            try:
                info = yf.Ticker(ticker).info
                summary = info.get("longBusinessSummary", "")
                if summary and len(summary) > 50:
                    descriptions[ticker] = summary
                    sources[ticker] = "Yahoo"
                    print(f"    {ticker}: Yahoo ({len(summary.split())} words)")
                else:
                    print(f"    {ticker}: NO DESCRIPTION FOUND")
            except Exception as e:
                print(f"    {ticker}: Error - {e}")
            time.sleep(0.5)
    
    print(f"\n  Results: {len(descriptions)}/{len(tickers)} tickers have descriptions")
    from collections import Counter
    src_counts = Counter(sources.values())
    for src, cnt in src_counts.most_common():
        print(f"    {src}: {cnt}")
    
    return descriptions, sources


def build_lexical_matrix(descriptions, tickers):
    """Build TF-IDF cosine similarity matrix."""
    print(f"\nBuilding TF-IDF lexical similarity matrix...")
    
    # Only use tickers that have descriptions
    valid = [t for t in tickers if t in descriptions]
    texts = [descriptions[t] for t in valid]
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,  # More features for richer 10-K text
        min_df=2,
        max_df=0.95,
    )
    tfidf = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf)
    
    lex_df = pd.DataFrame(sim, index=valid, columns=valid)
    
    # Stats
    upper = sim[np.triu_indices_from(sim, k=1)]
    print(f"  Matrix shape: {lex_df.shape}")
    print(f"  Avg similarity: {np.mean(upper):.4f}")
    print(f"  Min: {np.min(upper):.4f}, Max: {np.max(upper):.4f}")
    print(f"  Std: {np.std(upper):.4f}")
    
    return lex_df, valid


def main():
    print("=" * 60)
    print("Phase 4: 100 S&P 500 Stocks with 10-K Filing Analysis")
    print("=" * 60)
    
    # 1. Get tickers and sample 100
    all_tickers, all_sectors = get_sp500_tickers()
    selected, sel_sectors = sample_stocks(all_tickers, all_sectors, n=100)
    
    # 2. Download prices
    price_df = download_prices(selected)
    valid_tickers = list(price_df.columns)
    
    # Update sector map for valid tickers only
    sel_sectors = {t: sel_sectors.get(t, "Unknown") for t in valid_tickers}
    
    # 3. Get 10-K descriptions
    descriptions, sources = get_business_descriptions(valid_tickers)
    
    # Filter to tickers with both prices and descriptions
    final_tickers = [t for t in valid_tickers if t in descriptions]
    price_df = price_df[final_tickers]
    sel_sectors = {t: sel_sectors[t] for t in final_tickers}
    
    print(f"\n  Final dataset: {len(final_tickers)} tickers, {len(price_df)} days")
    
    # 4. Build lexical matrix
    lex_df, lex_tickers = build_lexical_matrix(descriptions, final_tickers)
    
    # Align everything to lex_tickers (tickers that passed TF-IDF)
    price_df = price_df[lex_tickers]
    sel_sectors = {t: sel_sectors[t] for t in lex_tickers}
    
    # 5. Save everything
    price_df.to_csv("data/sp500_100_prices.csv")
    print(f"\nSaved: data/sp500_100_prices.csv ({price_df.shape})")
    
    lex_df.to_csv("data/processed/lexical_matrix_100.csv")
    print(f"Saved: data/processed/lexical_matrix_100.csv ({lex_df.shape})")
    
    with open("data/processed/sector_map_100.json", "w") as f:
        json.dump(sel_sectors, f, indent=2)
    print(f"Saved: data/processed/sector_map_100.json ({len(sel_sectors)} tickers)")
    
    with open("data/raw/sp500_100_10k_texts.json", "w") as f:
        json.dump({
            "descriptions": descriptions,
            "sources": sources,
        }, f, indent=2)
    print(f"Saved: data/raw/sp500_100_10k_texts.json")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    from collections import Counter
    print(f"  Tickers: {len(lex_tickers)}")
    print(f"  Trading days: {len(price_df)}")
    print(f"  GICS sectors: {len(set(sel_sectors.values()))}")
    for sector, cnt in Counter(sel_sectors.values()).most_common():
        tks = [t for t, s in sel_sectors.items() if s == sector]
        print(f"    {sector} ({cnt}): {', '.join(tks)}")
    print(f"\n  Text sources:")
    for src, cnt in Counter(sources.values()).most_common():
        print(f"    {src}: {cnt}")
    
    upper = lex_df.values[np.triu_indices_from(lex_df.values, k=1)]
    print(f"\n  Lexical matrix stats:")
    print(f"    Mean similarity: {np.mean(upper):.4f}")
    print(f"    Std: {np.std(upper):.4f}")
    print(f"    Range: [{np.min(upper):.4f}, {np.max(upper):.4f}]")
    
    print("\nDONE!")


if __name__ == "__main__":
    main()
