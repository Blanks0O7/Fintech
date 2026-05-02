# Data and Preprocessing

## What data the project uses
The project combines two types of data:
1. Financial price data
2. Semantic business-description text

That is important in the defense because it shows the project is multimodal in a practical way.

## Financial price data
### What was used
- Daily price data for an S&P 500 stock universe.
- In the submitted code, the main pipeline uses `sp500_100_prices.csv` and then selects a subset for experiments.
- The notebook describes smart stock selection and risk-pool classification.

### Why S&P 500 stocks
Best answer:
- "I used a large-cap, liquid universe because I wanted realistic, stable assets with reliable price histories and public filings. That reduces noise from illiquidity and missing data."

### Why not small-cap or penny stocks
- Those assets can create artificial backtest results because of low liquidity, large gaps, and noisy pricing.
- That would make it harder to isolate the effect of the semantic penalty.

### Why daily frequency
- Daily data is a reasonable balance between signal and noise.
- Intraday data would be much noisier and less aligned with annual filing text.
- Monthly data would be too coarse for RL portfolio adaptation.

## Risk-pool construction
### What was done
- Stocks were classified into Safe, Neutral, and Risky pools using beta.
- Safe: beta below 0.8
- Neutral: beta between 0.8 and 1.2
- Risky: beta above 1.2

### Why beta was used
Best answer:
- "Beta is a simple and interpretable measure of systematic risk. It gives a direct way to partition the universe into defensive, market-like, and aggressive pools."

### Why not volatility only
- Volatility measures total variability, not market sensitivity.
- Beta is more aligned with how the dissertation defines systematic risk and market exposure.

### Why not sector-based pools
- Sector pools are easier to explain, but they are coarser.
- Beta pools support the finance argument more directly because they define risk exposure rather than industry label.

## Price preprocessing
### What was done
- Prices were transformed into returns.
- Rolling windows were used as observations.
- The EIIE structure uses normalized recent price history rather than raw levels.

### Why rolling windows
- RL decisions need state information, not just a single price.
- A rolling window gives short-term context on trend and movement shape.

### Why normalization matters
- Raw prices are not comparable across assets with different price levels.
- Normalization lets the model learn shape rather than scale.

## Text data
### What text source was used
- SEC EDGAR 10-K Item 1 business descriptions.
- In the codebase, text was gathered from Yahoo Finance summaries and SEC EDGAR fallback where needed.

### Why 10-K business descriptions
Best answer:
- "I needed a stable source of business-structure information. 10-K business descriptions are formal, regulated, and directly about what the company does."

### Why not news headlines
- News is more event-driven and noisy.
- This project wants structural similarity, not short-term sentiment.

### Why not earnings-call transcripts only
- They are useful, but more conversational and less standardized.
- 10-K business descriptions are better for a clean base similarity measure.

## TF-IDF and cosine similarity
### What was done
- Texts were cleaned.
- Stop words were removed.
- TF-IDF vectors were created.
- Cosine similarity was computed between company descriptions.
- This produced the lexical similarity matrix S.

### Why TF-IDF
Best answer:
- "TF-IDF is simple, transparent, and easy to defend. It captures which words are important in one document relative to the whole corpus."

### Why cosine similarity
- It measures directional similarity between text vectors.
- It is standard for document comparison.
- It works well when document length varies.

### Why not transformers or BERT embeddings
Best answer:
- "Transformer embeddings are a strong future direction, but TF-IDF is more interpretable and easier to audit. For a dissertation, that transparency is valuable."

### Stronger expert answer
- "TF-IDF trades semantic depth for interpretability and reproducibility. Since this dissertation's main claim is about integrating business similarity into RL, a transparent baseline representation was an appropriate first step."

## Why the matrix is useful
The matrix S is a map of business overlap.
- High value means two firms describe very similar businesses.
- Low value means their business descriptions are more different.

This matters because the lexical penalty uses both similarity and portfolio weight.

## Why the matrix is held fixed
### What was done
- The semantic matrix is computed once and then used during training and evaluation.

### Why this is defendable
- Business models change slowly compared with daily prices.
- A fixed matrix isolates the effect of the semantic structure more clearly.

### Limitation
- It cannot adapt quickly to new sector-level shocks.
- This is why 10-Q updates are proposed in future work.

## Data-quality questions experts may ask
### "How did you handle missing data?"
Answer:
- The code filters and forward/backward fills where appropriate, and the stock selection step favors cleaner data. The goal was to keep a stable universe with good coverage.

### "Why not use OHLCV features everywhere?"
Answer:
- The submitted implementation mainly uses price-based windows and related return features. This keeps the state compact and aligned with the EIIE-style setup.

### "Why use both price data and text data?"
Answer:
- Price data is necessary for return generation and trading behavior. Text data adds structural information about business overlap that price alone may miss.

## Best short summary
- "The data design reflects the central thesis: prices tell us how firms move, while 10-K business descriptions tell us what firms are. The model needs both."
