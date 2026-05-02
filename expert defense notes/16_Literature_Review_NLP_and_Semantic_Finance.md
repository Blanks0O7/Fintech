# Literature Review: NLP and Semantic Finance

## Core message
This part of the literature review justifies why company text can be useful for portfolio construction, especially for diversification rather than only short-term prediction.

## Why text matters in finance
- Text can contain information that prices and ratios do not fully reveal.
- Business descriptions capture persistent firm characteristics.
- Similar text can imply similar economic exposures.

## Loughran and McDonald
### Why important
- They show that financial language is domain-specific.
- Generic sentiment dictionaries are often misleading.

### Why this supports your work
- It justifies treating financial text as a serious data source rather than casual sentiment material.

## Tetlock
### Why important
- It is a classic demonstration that text carries market-relevant signal.

### Why your dissertation still differs
- Your dissertation is not mainly about sentiment forecasting.
- It is about semantic similarity and diversification structure.

## Hoberg and Phillips
### Why they are especially important
- They show text-based business similarity has real economic meaning.
- This is the strongest direct support for your semantic penalty idea.

### Best answer
- "Hoberg and Phillips matters most because it links business-description similarity to economically meaningful firm relationships. That is the bridge from NLP to portfolio diversification."

## Cohen, Malloy, and Nguyen
### Why important
- They support the view that filing text contains persistent information.
- That helps justify using 10-K text for structural signals rather than only transient events.

## Mohseni et al. and lexical diversification
### Why important
- They provide a direct semantic-diversification idea.
- Your dissertation extends that idea into a trainable RL reward.

### Best answer
- "Their work inspired the semantic diversification concept, but my dissertation moves it into sequential decision-making rather than keeping it as a static optimization idea."

## Why business text is better than ticker count for diversification
- Owning many names is not the same as owning different business exposures.
- Text similarity gives a continuous signal of operational overlap.

## Why business text can be better than sector labels
- Text can capture overlap within a sector and across sectors.
- Sector labels are too coarse and categorical.

## Why 10-K text was a sensible starting point
- Standardized
- Regulated
- Publicly available
- Business-focused
- Lower noise than social media or headlines

## Why semantic finance literature supports your contribution
- It says text is useful.
- Your dissertation tests one specific use: semantic diversification inside RL.

## Common expert challenge
### "Does text similarity really imply return co-movement?"
Answer:
- Not perfectly in every case. But the literature suggests it is a meaningful proxy for economic overlap, and the dissertation tests whether using that proxy improves portfolio behavior.

## Final one-line summary
- "The NLP literature gives the missing information source that price-only portfolio models do not have: a representation of what firms actually do."
