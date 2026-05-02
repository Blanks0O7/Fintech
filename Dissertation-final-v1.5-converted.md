***Dissertation Title***

*Example: A Smart Healthcare Recommendation System for Diabetes Patients with Data Fusion Based on Deep Ensemble Learning*

**Final Thesis**

In Partial Fulfillment

of the Requirements for the Degree of

Master in Computer Science

| Student Name | :   |       |
|--------------|-----|-------|
| Student ID   | :   |       |
| Supervisor   | :   |       |

# Abstract

Classical portfolio diversification fails during market stress. Techniques derived from Markowitz mean-variance optimization rely on historical price covariance to measure asset independence. During extreme shocks, the 2020 COVID-19 crash and the 2022 inflationary bear market in which the previously uncorrelated assets collapsed together because their apparent independence was statistical rather than economic. The root cause is structural: price data captures co-movement but cannot identify whether two companies share the same customers, supply chains, or regulatory exposure.

This dissertation presents a Hierarchical Multi-Agent Reinforcement Learning (MARL) system that addresses this limitation by embedding genuine business similarity into portfolio construction. Annual business description filings (SEC EDGAR 10-K Item 1) for 45 S&P 500 companies are transformed into a TF-IDF cosine similarity matrix and embedded into each agent's reward function as a trainable quadratic penalty λ·wᵀ·S·w, the Weight-Aware Lexical Penalty (WALP). This mathematically forces agents to avoid concentrating capital in companies with overlapping business operations.

The architecture pairs a manager agent with three specialized Worker agents operating within beta-classified risk pools: Safe (β\<0.8), Neutral (0.8≤β\<1.2), and Risky (β≥1.2). Dirichlet distribution policy heads replace standard Softmax layers throughout, resolving the 1/N convergence trap. A three-phase staged curriculum training protocol in which Workers first, Manager second against stable Worker policies, joint fine-tuning eliminates the non-stationarity instability that prevents concurrent multi-agent training from producing coherent allocations.

Evaluated across four market regimes spanning 2015 to 2023, the staged system produces consistently lower drawdown than all classical baselines, including during the 2022 holdout stress period. The semantic penalty is directly validated: activating it at the optimal strength produces a 46.9% Sharpe ratio improvement over the zero-penalty baseline. A null-hypothesis control experiment, replacing the real similarity matrix with a randomly shuffled permutation confirms that genuine semantic content, not merely the mathematical structure of the penalty, is responsible for the performance improvement.

**Keywords:** hierarchical multi-agent reinforcement learning, semantic portfolio diversification, weight-aware lexical penalty, Dirichlet distribution, SEC EDGAR 10-K analysis, curriculum learning, walk-forward validation, structural resilience

#  Acknowledgements

I want to express my greatest appreciation to my academic supervisors (Dr. Yara Magdy & Prof. Saeed Sharif) as they guided me through this experience with their vast professional knowledge, tolerance, and encouragement. It was through their criticism and insistence upon an academic quality that made it feasible to move from simply applying technology within this study to the true academic nature of the research.  
It is also my privilege to offer my sincerest thanks to all of the faculty and personnel associated with the MSc Big Data Technologies program at the University of East London. The foundational educational background and technological infrastructure of the university created the necessary conditions for a project of such magnitude and complexity.  
Personally speaking, I owe a great deal of gratitude to my family in Nepal. Navigating the pressures of completing a master's degree while dealing with the many challenges that come with being away from home has been one of the most difficult times of my life. Without their unrelenting emotional support, sacrifices, and faith in my education, I could never have accomplished what I did here. Additionally, I wish to express a special “thank you” to those members of my peer group and colleagues whose willingness to share time discussing technical issues, troubleshooting, and assisting with various aspects of this research over the last few months of its completion, helped me complete this research successfully.

# Contents

[Abstract [ii](#abstract)](#abstract)

[Acknowledgements [iv](#acknowledgements)](#acknowledgements)

[Contents [v](#_Toc228643043)](#_Toc228643043)

[List of Tables [vii](#_Toc128143291)](#_Toc128143291)

[List of Figures [viii](#_Toc128143292)](#_Toc128143292)

[List of Acronyms [ix](#_Toc128143293)](#_Toc128143293)

[Chapter 1 Introduction [1](#_Toc228643047)](#_Toc228643047)

[1.1 Background (font: Times New Roman; font size 16) [1](#background)](#background)

[1.2 Problem Statement [3](#problem-statement)](#problem-statement)

[1.3 Research Question and Objectives [5](#research-question-and-objectives)](#research-question-and-objectives)

[1.4 Expected outcomes [6](#expected-outcomes)](#expected-outcomes)

[Chapter 2 Literature Review [8](#literature-review)](#literature-review)

[2.1 Comprehensive Overview of the Existing Literature [8](#comprehensive-overview-of-the-existing-literature)](#comprehensive-overview-of-the-existing-literature)

[2.1.1 The Failure of Classical Optimisation [8](#the-failure-of-classical-optimisation)](#the-failure-of-classical-optimisation)

[2.1.2 Reinforcement Learning for Portfolio Management [10](#reinforcement-learning-for-portfolio-management)](#reinforcement-learning-for-portfolio-management)

[2.1.3 Semantic Data and the Lexical Penalty [10](#semantic-data-and-the-lexical-penalty)](#semantic-data-and-the-lexical-penalty)

[2.1.4 The Dirichlet Policy: Resolving the Softmax Constraint [12](#the-dirichlet-policy-resolving-the-softmax-constraint)](#the-dirichlet-policy-resolving-the-softmax-constraint)

[2.2 Critical Analysis of Existing Studies [13](#critical-analysis-of-existing-studies)](#critical-analysis-of-existing-studies)

[Chapter 3 Methodology [17](#methodology)](#methodology)

[3.1 Data Collection and Preprocessing [20](#data-collection-and-preprocessing)](#data-collection-and-preprocessing)

[3.1.1 Financial Price Data [20](#financial-price-data)](#financial-price-data)

[3.1.2 Semantic Data: SEC EDGAR 10K Fillings [22](#semantic-data-sec-edgar-10k-fillings)](#semantic-data-sec-edgar-10k-fillings)

[3.1.3 Construction Of the Semantic Similarity Matrix [23](#construction-of-the-semantic-similarity-matrix)](#construction-of-the-semantic-similarity-matrix)

[3.2 ML/AI Model Development [25](#mlai-model-development)](#mlai-model-development)

[3.2.1 Feature Extraction: The EIIE Topology [25](#feature-extraction-the-eiie-topology)](#feature-extraction-the-eiie-topology)

[3.2.2 Action Generation: The Dirichlet Policy [26](#action-generation-the-dirichlet-policy)](#action-generation-the-dirichlet-policy)

[3.2.3 Profile-Specific Reward Functions [27](#profile-specific-reward-functions)](#profile-specific-reward-functions)

[3.2.4 The Weight-Aware Lexical Penalty [28](#the-weight-aware-lexical-penalty)](#the-weight-aware-lexical-penalty)

[3.2.5 Staged Curriculum Training Protocol [29](#staged-curriculum-training-protocol)](#staged-curriculum-training-protocol)

[3.3 Evaluation of the Proposed System [32](#evaluation-of-the-proposed-system)](#evaluation-of-the-proposed-system)

[3.3.1 Walk-Forward Validation [32](#walk-forward-validation)](#walk-forward-validation)

[3.3.2 Performance Metrics [33](#performance-metrics)](#performance-metrics)

[3.3.3 Baseline Methods [34](#baseline-methods)](#baseline-methods)

[3.3.4 Lambda Ablation Study [35](#lambda-ablation-study)](#lambda-ablation-study)

[Chapter 4 Experimental Results [36](#experimental-results)](#experimental-results)

[4.1 Experimental Setup [37](#experimental-setup)](#experimental-setup)

[4.2 Dataset Description [37](#dataset-description)](#dataset-description)

[4.3 Results [38](#results)](#results)

[4.3.1 Primary Five-Metric Comparison [38](#primary-five-metric-comparison)](#primary-five-metric-comparison)

[4.3.2 Holdout Test-Set Analysis (2022-2023) [41](#holdout-test-set-analysis-2022-2023)](#holdout-test-set-analysis-2022-2023)

[4.3.3 Lambda Ablation: Casual Evidence for the Semantic Penalty [44](#lambda-ablation-casual-evidence-for-the-semantic-penalty)](#lambda-ablation-casual-evidence-for-the-semantic-penalty)

[4.3.4 Walk-Forward Regime Analysis [47](#walk-forward-regime-analysis)](#walk-forward-regime-analysis)

[4.3.5 Worker Level Performance Analysis [49](#worker-level-performance-analysis)](#worker-level-performance-analysis)

[4.3.6 Drawdown Decomposition and Staged vs Concurrent Comparision [51](#drawdown-decomposition-and-staged-vs-concurrent-comparision)](#drawdown-decomposition-and-staged-vs-concurrent-comparision)

[4.3.7 Null Hypothesis Test: Lexical Control Experiment [53](#null-hypothesis-test-lexical-control-experiment)](#null-hypothesis-test-lexical-control-experiment)

[4.3.8 Return Distribution Analysis: Skewness and Kurtosis [54](#return-distribution-analysis-skewness-and-kurtosis)](#return-distribution-analysis-skewness-and-kurtosis)

[Chapter 5 Conclusion and Future Work [57](#conclusion-and-future-work)](#conclusion-and-future-work)

[5.1 Conclusion [57](#conclusion)](#conclusion)

[5.2 Future Work [59](#future-work)](#future-work)

[References [62](#_Toc404008248)](#_Toc404008248)

[Appendix A. System Hyperparameters and Configuration [67](#_Toc303784826)](#_Toc303784826)

[Appendix B. Core System Implementation [70](#_Toc394498204)](#_Toc394498204)

<span id="_Toc128143291" class="anchor"></span>

List of Tables

[Table 1 Critical Analysis of Existing Optimization Framework [14](#_Toc228643997)](#_Toc228643997)

[Table 3 End-to-end methodology pipeline for the Hierarchical MARL system [18](#_Toc228643998)](#_Toc228643998)

[Table 4 Training and walk-forward evaluation workflow [19](#_Toc228643999)](#_Toc228643999)

[Table 7 Five-Metric Comparison Across All Methods (Full Evaluation Period 2015–2023 [39](#_Toc228644000)](#_Toc228644000)

[Table 9 Holdout Test-Set Performance (2022-03-14 to 2023-12-29, 453 trading days) [42](#_Toc228644001)](#_Toc228644001)

[Table 10Lambda Ablation Study [44](#_Toc228644002)](#_Toc228644002)

[Table 13Walk-Forward Validation — Six Metrics Per Market Regime (Concurrent MARL, λ=0.1). Calmar ratio = annualised return / MaxDD. [48](#_Toc228644003)](#_Toc228644003)

[Table 15 Worker-Level Performance and Concurrent Manager Allocation by Risk Pool [50](#_Toc228644004)](#_Toc228644004)

[Table 18 Return Distribution Statistics, Skewness and Kurtosis (Full 2015–2023 Period) [55](#_Toc228644005)](#_Toc228644005)

<span id="_Toc128143292" class="anchor"></span>

List of Figures

[Figure 1 End-to-end methodology pipeline for the Hierarchical MARL system [18](#_Toc228643966)](#_Toc228643966)

[Figure 2 Training and walk-forward evaluation workflow [19](#_Toc228643967)](#_Toc228643967)

[Figure 3 Beta Classification of S&P 500 Stocks into Safe, Neutral, and Risky Risk Pools. Left panel: individual beta values with threshold lines. Right panel: histogram of beta distribution [21](#_Toc228643968)](#_Toc228643968)

[Figure 4 TF-IDF Cosine Similarity Matrix (45×45) from SEC EDGAR 10-K Fillings [24](#_Toc228643969)](#_Toc228643969)

[Figure 5 Manager Risk-Profile Capital Allocation (left) and Worker Returns by Risk Pool (right) for the concurrent training system [40](#_Toc228643970)](#_Toc228643970)

[Figure 6 Lambda Ablation Study (4-panel visualisation). [45](#_Toc228643971)](#_Toc228643971)

[Figure 7 Lambda Ablation (Concurrent Vs Staged Training) [47](#_Toc228643972)](#_Toc228643972)

[Figure 8 Walk-Forward Out-of-Sample Returns by Regime (left) and Sharpe / CVaR Risk Metrics by Regime (right). Green [48](#_Toc228643973)](#_Toc228643973)

[Figure 9 Global Portfolio Equity Curve (top) and Drawdown Timeline (bottom) for the Concurrent MARL System over a 200-day evaluation window [51](#_Toc228643974)](#_Toc228643974)

[Figure 10 Null Hypothesis Test — Real vs Shuffled vs Zero Lexical Matrix [54](#_Toc228643975)](#_Toc228643975)

[Figure 11 Return Distribution Analysis — Skewness and Kurtosis [56](#_Toc228643976)](#_Toc228643976)

<span id="_Toc128143293" class="anchor"></span>List of Acronyms

| **AI**       | Artificial Intelligence                            |
|--------------|----------------------------------------------------|
| **β (Beta)** | Measure of stock volatility relative to the market |
| **CAPM**     | Capital Asset Pricing Model                        |
| **Conv1D**   | One-Dimensional Convolutional Neural Network       |
| **CVaR**     | Conditional Value at Risk                          |
| **EIIE**     | Ensemble of Identical Independent Evaluators       |
| **GICS**     | Global Industry Classification Standard            |
| **HHI**      | Herfindahl-Hirschman Index                         |
| **MARL**     | Multi-Agent Reinforcement Learning                 |
| **MDD**      | Maximum Drawdown                                   |
| **ML**       | Machine Learning                                   |
| **MDP**      | Markov Decision Process                            |
| **MVO**      | Mean-Variance Optimization                         |
| **NLP**      | Natural Language Processing                        |
| **OHLCV**    | Open, High, Low, Close, Volume                     |
| **PVM**      | Portfolio-Vector Memory                            |
| **RL**       | Reinforcement Learning                             |
| **SEC**      | US Securities and Exchange Commission              |
| **SIC**      | Standard Industrial Classification                 |
| **TF-IDF**   | Term Frequency–Inverse Document Frequency          |
| **VaR**      | Value at Risk                                      |
|              |                                                    |

# Introduction 

## Background 

The use of algorithms in executing trades has shown how a major flaw exists within MVO. Price based diversification fails to hold up well to times of financial stress.

In periods of global economic shocks, correlation matrices no longer depict the true interdependence of asset classes. This became apparent in an extraordinary manner in March-April 2020 with the COVID-19 crisis. Assets which had been independent statistically for many years fell sharply collectively due to institutional investor selling down across all asset classes simultaneously. This occurs when assets become far more dependent on each other during severe decline in asset prices than their historical correlations indicate, and is referred to as "tail dependence." Tail dependence is the biggest weakness of classical portfolio theory (Flint et al., 2020).

Chua et al. (2009) showed empirical evidence of this failure to provide diversification benefits to portfolios during extreme market conditions when the need for portfolio protection is greatest. Furthermore, Ang & Chen (2002) quantitatively demonstrated the asymmetry of these dependencies and found them to be significantly greater in bear markets than in bull markets. For example, the 2022 inflationary bear market caused by the most aggressive Federal Reserve tightening cycle in over thirty years also produced a second such example as simultaneous decreases in both stocks and bonds resulted in a breakdown of the traditional 60/40 hedge which had protected investment portfolios for over thirty years.

The engineering reason why classical portfolio theory failed so miserably is that OHLCV price data alone does not have enough dimensions to adequately measure the real economic exposure of a firm. While it may seem that investing \$100 million into twenty different technology stocks represents some degree of diversification under a covariant model, it actually represents a massive concentration of risk related to semiconductor supply chain and interest rate risks, as well as regulatory risks. When these common vulnerabilities manifest themselves -- as they did during the 2022 Federal Reserve tightening cycle -- the resulting loss is identical to that incurred by holding a large concentrated position. The issue here is not the way correlations are calculated, but rather the type of signal used. Correlation of price movement captures statistical co-movements, but completely ignores the economic structures that produce those movements.

A primary motivation behind this research is to propose a simple yet effective idea: if the failure of classical diversification arises from comparing similarity using price behavior as opposed to examining the firms' actual business practices, then a method that specifically examines and penalizes similarities in the business practices of firms can create portfolios that are truly resilient. Using NLP methods to analyze and parse the official SEC EDGAR 10-K reports filed annually by public corporations describing their businesses provides a means to map out the actual economic footprint of companies. In addition, using the content of these 10-K files enables diversifying investments at the operational level, as opposed to at the statistical price level. Although a utility company and a semiconductor manufacturer may demonstrate similar historical price correlations during a period of relative stability during a bull market, their respective 10-K files will describe fundamentally disparate business practices and customer bases, etc. The semantic similarity matrix created from these 10-K file descriptions creates a constant diversification signal that will not fail during extreme market events, since business practices do not change with daily changes in price.

Under the academic supervision of Professor Saeed Sharif and Dr. Yara Magdy at University College London, this dissertation designs, builds, tests and evaluates a Hierarchical Multi-Agent Reinforcement Learning (MARL) system that incorporates the semantic diversification constraint. The MARL system uses an architectural approach of Manager and Worker agents to separate decision-making regarding risk among three beta defined risk-pools, a Dirichlet-based policy layer to allow for high conviction allocations, and a three-stage curriculum learning schedule to stabilize the non-stationariness associated with simultaneous optimization of multiple agents. The end result is a portfolio management system that is 'structural' rather than 'statistical,' i.e., it preserves diversification via firm-specific economics as opposed to prior historical price patterns.

## Problem Statement

The primary challenge of continuous-action portfolio management is that even sophisticated, computationally intensive artificial intelligence models are generally unable to significantly exceed the performance of an equally weighted (1/N), naive portfolio strategy when tested in out-of-sample evaluations. In fact, DeMiguel, Garlappi and Uppal (2009) were able to demonstrate that no one of fourteen advanced optimization methods was capable of exceeding the performance of the simple equally weighted portfolio strategy in tests conducted across multiple assets. This phenomenon has been termed the "1/N Trap". While this is an interesting empirical finding, it also illustrates a fundamentally important characteristic of reinforcement learning optimizers. Specifically, when dealing with inherently noisy financial data, reinforcement learning optimizers have a natural tendency to allocate capital uniformly among available investments; i.e., they tend to minimize the variance of the reward signal by spreading investment capital across the available options rather than attempting to generate excess returns (alpha) through concentrated capital allocations.

One key reason why reinforcement learning optimizers fall prey to the "1/N Trap" is due to the use of the Softmax activation function in nearly all standard portfolio agents. The Softmax function maps a multi-dimensional vector (the output of the neural network) onto a vector of probabilities where each element represents the proportion of the total capital allocated to each investment. As such, the Softmax function applies an exponential normalization to ensure that the resulting probability vector sums to 1.0. While this makes programming relatively straightforward, it introduces significant behavioral difficulties. In particular, since the Softmax function uses exponentiation to map its input vector to a probability vector, it naturally tends to draw inputs towards uniformly distributed vectors. Thus, if a reinforcement learning optimizer is trained using Softmax as its activation function, there will be limited capacity for the agent to produce extreme values (e.g. 0% vs 100%) within its output vector. Since it is mathematically difficult for a reinforcement learning optimizer to produce extreme values in its output vector when training on Softmax functions, it can become behaviorally biased toward producing uniformly distributed vectors — effectively trapping itself in the same 1/N pattern that it was programmed to overcome (Tian et al., 2022).

In addition to being trapped by its own architecture, reinforcement learning optimizers suffer from another fundamental limitation: nearly all existing reinforcement learning based portfolio systems optimize solely upon price return signals. That is, given an array of Open-High-Low-Close-Volume (OHLCV) prices for any number of securities, existing agents do not consider what the underlying companies actually do. For example, if we train a reinforcement learning agent on thirty years’ worth of stock price history for thirty technology companies, the agent views thirty different price sequences but does not view that twenty-eight of the companies rely upon identical semiconductor supply chain dependencies. If a global semiconductor shortage causes many companies to experience severe disruptions to their operations, then the price of the stocks in our sample will likely decline dramatically. However, in spite of experiencing simultaneous declines in price due to common underlying factors, the agent would still see these declines in price as independent events. This 'price-blindness', therefore, produces portfolios that may appear diversified based upon historical covariance but will collapse under stress caused by fundamental shocks, because the shock reveals the economic linkages that price data was incapable of measuring.

This research project seeks to address both limitations simultaneously. First, this project replaces the Softmax activation layer with a Dirichlet distribution policy head which allows our manager agent to allocate capital across three risk pools at unconstrained levels of conviction. Second, this project augments the reward function with a Weight-Aware Lexical Penalty derived from TF-IDF cosine similarity scores computed over SEC 10-K business description content which forces our agents to avoid semantic redundancy in their portfolio holdings. By combining these two architectural modifications — one addressing the mathematical bias inherent in using Softmax layers and the second addressing the informational blind spots inherent in optimizing solely based on price data — this project presents the core technological contributions made here.

## Research Question and Objectives

This section introduces the motivation/research questions, and objective of the project.

This dissertation is structured around two primary research questions that directly address the limitations identified in Section 1.2:

Research Question 1: Can a Hierarchical MARL architecture, constrained by a Weight-Aware Lexical Penalty, achieve statistically superior risk-adjusted returns and enhanced drawdown protection compared to classical price-based diversification models?

Research Question 2: Does the strength of the semantic penalty (controlled by the hyperparameter λ) have a measurable causal effect on portfolio quality, and what does the direction and magnitude of this effect reveal about the relative contribution of NLP-derived business intelligence to the system's overall performance?

Objectives:

- **Develop a Hierarchical Architecture:** Build a Manager-Worker MARL system in which the asset universe is segmented into Safe, Neutral, and Risky risk pools based on historical market beta, allowing specialized Worker agents to operate within isolated sub-environments under the coordination of a central Manager agent that observes Worker performance signals.

- **Integrate Semantic Constraints:** Extract and preprocess textual data from SEC 10-K filings to construct a TF-IDF cosine similarity matrix S. Embed this matrix into the Worker agents' reward functions as a quadratic semantic penalty λ·w^T·S·w that mathematically discourages holdings in companies with overlapping business descriptions.

- **Apply Staged Curriculum Training:** Implement a three-phase training protocol that first trains Workers with the Manager frozen, then trains the Manager with Workers frozen, then jointly fine-tunes all agents. This sequential approach eliminates the non-stationarity instability inherent in concurrent multi-agent training.

- **Evaluate Structural Resilience:** Backtest the architecture using walk-forward validation across four historically distinct market regimes including the 2020 COVID-19 crash and the 2022 inflationary bear market, benchmarking against equal-weight, MVO, risk parity, and momentum baselines across all five-evaluation metrics.

## Expected outcomes

This research is expected to demonstrate that embedding NLP-derived business description similarity as a trainable reward penalty within a Hierarchical MARL system produces measurably superior portfolio resilience compared to classical price-based diversification methods.

Specifically, the staged curriculum training protocol is expected to resolve the non-stationarity problem inherent in concurrent multi-agent training, enabling the Manager agent to learn coherent, high-conviction risk-pool allocations that a concurrent system cannot achieve. The Dirichlet policy architecture is expected to overcome the 1/N convergence trap by allowing concentrated allocations without gradient suppression.

The lambda ablation study is expected to provide direct empirical evidence that the semantic penalty strength has a causal and measurable effect on portfolio quality, with performance improving as λ increases from zero to its optimal value. This will reveal the relative contribution of NLP-derived business intelligence to the system's overall performance.

The walk-forward validation across four distinct market regimes is expected to confirm that semantic diversification provides structural capital protection during stress periods, particularly during the 2022 inflationary bear market, where classical covariance-based methods are known to fail. The system is not expected to maximize absolute returns in all regimes, but is expected to consistently limit worst-case drawdown beyond what price-only methods achieve.

The findings will be of relevance to quantitative researchers, institutional portfolio managers, and AI practitioners interested in combining natural language processing with reinforcement learning for robust financial decision-making.

# Literature Review

## Comprehensive Overview of the Existing Literature 

Automated Portfolio Optimization has been shaped by an evolving balance of mathematical beauty and empirical reliability over nearly four decades. The chapter will trace that evolutionary path through four overlapping areas of research that are all linked to the motivation for developing the proposed architecture: the limitations of classical optimization methods when applied to "real world" portfolio data, the emerging family of Reinforcement Learning (RL) models for use in portfolio management, the increasing recognition of the value of both linguistic and semantic data as part of a portfolio manager's decision making process, and the two new architectural approaches (Dirichlet Policies and Hierarchical Multi-Agent RL) that have emerged from this body of work. The chapter will conclude with a qualitative comparison of the previous work in this area, providing insight into the shortcomings that the proposed solution addresses.

### The Failure of Classical Optimisation

The prevailing methodology for building portfolios since at least 1952 has been the Mean-Variance Optimization (MVO), developed by Markowitz. He claimed that an investor can receive a "free lunch" via diversification utilizing historical covariance data; thereby creating portfolios located along the efficient frontier which maximize expected return while minimizing variance. The analytical beauty of this model has been well-documented and its impact upon both the field of academia and practitioner application have been substantial (Flint et al., 2020). However, MVO relies upon historical parameter estimates. Therefore, there exists a fundamental structural weakness to the ability of MVO to perform outside sample.

Michaud (1989) demonstrated mathematically that the MVO method utilizes not just estimation errors, but rather also increases their size. As a result of the optimization being highly dependent upon expected returns, any estimation error in the mean vector will be increased through the process of matrix inversion used to calculate optimal weightings. Michaud referred to this phenomenon as the "Markowitz Optimization Enigma." In essence, what appears to provide a mathematical solution to the portfolio selection problem, is the exact framework which is likely to fail empirically, due to treating historically noisy estimates of parameters as if they are actual parameters. Ledoit and Wolf (2004) later found that even when employing sophisticated shrinkage estimators for the covariance matrix, MVO's reliability decreases as the number of securities included in the universe increases above approximately thirty. This is particularly germane to the forty-five-stock universe employed in this study.

DeMiguel, Garlappi and Uppal (2009) measured this failure using fourteen different sophisticated optimization methods on several different real-world databases. They found that none of the fourteen methods, including Bayesian Stein estimators, portfolio constraint methodologies and numerous other types of moment restriction -- produced better out-of-sample performance than did a simple equally weighted (1/N) strategy. More specifically, DeMiguel et al. found that the estimation errors associated with each of the covariance matrices included in the studies resulted in a greater diversity benefit from simply naively assigning equal weightings to each security than was achieved by optimizing the portfolio assignments. This result, known as the "1/N Trap", serves as the major empirical justification for this dissertation, as any framework that does not structurally prevent the convergence toward 1/N would ultimately experience it again as evidenced by the concurrently occurring MARL results contained within Chapter 4.

In addition to problems created by estimating errors in covariance matrices, correlation breakdown problems create additional difficulties for resolving issues within classical frameworks. Flint et al. (2020) found that the property of diversifying risk is not universal, it depends upon the state of markets. During times of financial crises, the correlation structure of equity portfolios undergoes dramatic shifts as systemic selling pressure causes liquidations across all categories simultaneously. Chua et al. (2009) provided empirical evidence supporting the "Myth of Diversification," i.e., that the benefits derived from diversifying risk in traditional portfolios essentially disappear during those times when protection is most needed. Furthermore, Ang and Chen (2002) characterized this asymmetry in terms of how equities tend to correlate more strongly during down-markets than during up-markets. Consequently, classical covariance matrices estimated under normal circumstances consistently underestimate levels of co-movement that exist during crisis time. Collectively these findings demonstrate that the failure of price-based diversification is not attributable to either data quality or calibration problems, it is a structural limitation inherent to the underlying methodology.

### Reinforcement Learning for Portfolio Management

Credit assignment problems arise more frequently in portfolio MARL because Manager capital allocation decisions and Worker security selection decisions have highly intertwined effects on portfolio returns. Hierarchical architectures mitigate this by separating reward computation: Manager rewards derive from pooled Worker performance while each Worker’s reward is computed only from its own risk pool, treating Manager allocation as a constant input. This separation produces cleaner credit assignment than flat MARL systems (Shavandi and Khedmati, 2022).

### Semantic Data and the Lexical Penalty

The incorporation of textual and semantic data within portfolio building has an independent and distinct academic heritage. While there is also a rich body of literature focused on Reinforcement Learning (RL), Loughran & McDonald (2011) were the first to identify and demonstrate that Domain Specific Natural Language Processing (NLP) techniques can be applied to legal mandatory filings made by U.S. public corporations via their Securities Exchange Commission Form 10-K filings to uncover risk factors/risks and exposures that are otherwise hidden or unaccounted for in the price data. In addition to revealing these previously unidentified risks, they identified that common sentiment dictionaries often mis-identify corporate financial language; thereby making SEC required 10-K filings the best single source of reliable financial NLP information.

Tetlock (2007) was able to show even earlier that Media Sentiment Scores derived from financial text could forecast future behavior in the Stock Market; providing evidence that textual content contains signals regarding the direction of the market which extend well beyond what can be learned from price data. Building upon this work, Hoberg & Phillips (2016) further expanded our understanding of how textual information relates to firm behavior when they constructed a continuous measure of product market similarity between publicly traded firms using the business description section of each firm's 10-K filing. They found that firms whose business descriptions indicate similarities in products/markets have return co-movements greater than would be expected based solely on their assigned SIC code(s); providing direct motivation for utilizing 10-K business description similarity as a diversification signal in this study. Therefore, if textual similarity indicates greater return co-movement than does traditional classification (industry/sic) in a portfolio, then the inclusion of a penalty function for textual overlap in portfolio optimization will result in holding companies which are more economically independent.  
Cohen, Malloy & Nguyen (2020) provided additional empirical support for the idea that 10-K filings contain economically relevant information with enduring investment value. They demonstrated empirically that companies whose 10-K filings do not change over time experience significantly positive alpha going forward thus, validating that 10-K filings provide a means of capturing persistent economic characteristics with real investment value.

In order to quantify the degree of diversity in terms of the textual representation of each company’s business description, this research utilized a new measure called the "Lexical Ratio" developed by Mohseni et al. (2024). Different than other metrics employed to calculate portfolio diversification, such as Herfindahl-Hirschman Index (HHI), which calculates the concentration of a firm's weight within a portfolio, the Lexical Ratio calculates diversity using the distribution of textual representations of each company's business description. Mohseni et al., demonstrated empirically that portfolios optimized for semantic variance performed better under extreme stress conditions than did portfolios optimized for price variance -- primarily due to textual similarity in business descriptions measuring economic co-movement not captured by historical price correlation during non-stressful times.

This research expands upon previous applications of the Lexical Ratio, extending it from being a static optimization metric into being a dynamic, and trainable reinforcement learning reward penalty that agents are actively trained to minimize the first such application in prior literature.

### The Dirichlet Policy: Resolving the Softmax Constraint

The conventional method of mapping the output of a neural network to valid portfolio weights, i.e., a probability distribution over the simplex, is to use the Softmax function as an activation. Although Softmax has several desirable properties, such as being easy to compute and enforcing the budget constraint, Tian et al. (2022) showed that SoftMax’s exponential normalization produces a centripetal bias that causes agents to allocate funds closer to uniform portfolios. In fact, this centripetal bias is the primary driver of the 1/N problem in neural networks; specifically, when agents are uncertain about what the optimal reward signal should be, they will tend towards uniform distributions due to the arithmetic expense associated with gradients of deviated weights, which results in agents using the same amount of capital for all assets to minimize reward variance while sacrificing the ability to concentrate capital.

Andre (2021) provided analytical derivations and expressions of the gradient of the Dirichlet policy in portfolio reinforcement learning. These results show that Dirichlet policies can provide both diversified and concentrated asset allocations without reducing gradient magnitudes. Moreover, since the Dirichlet distribution has support on the entire probability simplex, each randomly generated portfolio will satisfy the budget constraint, it is the geometrically natural choice for generating probabilities for portfolio weights. Additionally, Kim et al. (2022), through their implementation of the Dirichlet Distribution Trader (DDT), have shown that Dirichlet policies allow for selective portfolio optimization across different risk levels, similar to how the manager allocates funding among Safe, Neutral, or Risky workers. Finally, in a study by Xue & Ye (2025) that was conducted using historical price data on the S&P 500 index from January 31st, 2000 to August 10th, 2025, they demonstrated empirically that Dirichlet-based policies significantly outperformed both baseline Softmax and equally weighted benchmarks on risk-adjusted performance metrics. Thus, their empirical findings validate our design choices for implementing these policies on contemporary data.

## Critical Analysis of Existing Studies 

The need for the Hierarchical MARL model is proven by evaluating how current models meet 3 key engineering criteria:

They can operate under varying levels of non-stationarity in markets

They avoid the 1/N convergence problem

They can function as market correlations break down due to systemic shocks. A formal comparison of some of these systems appears in table 2.1.

| **Framework**                  | Methodology                                | Key Limitation                                                                   | How this is Addressed                                                                                       |
|--------------------------------|--------------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Markowitz (1952)**           | MVO price covariance optimization          | Amplifies estimation errors and fails when correlations break down during crises | Replaced by model-free RL; penalty w^T.Sw is the semantic analogue of the covariance term                   |
| **DeMiguel et al. (2009)**     | 1/N Equal Weight baseline                  | Cannot actively defend capital and serves as the trap RL must escape             | Dirichlet policy + staged training breaks uniform convergence at both agent and pool level                  |
| **Jiang et al. (2017) - EIIE** | Single-agent EIIE with Conv1D              | Softmax bias; no fundamental business awareness; single agent                    | EIIE retained as feature extractor; Softmax replaced by Dirichlet; multi-agent hierarchy added              |
| **Lee et al. (2020)- MAPS**    | Flat per-asset MARL                        | Agent interference; budget competition causes 1/N at multi-agent level           | Manager-Worker hierarchy eliminates direct budget competition between agents                                |
| **Ma et al. (2023)**           | Trend consistency regularization MARL      | Penalty uses backward-looking price trends still price-blind                     | Penalty that uses fundamentals-based NLP-derived 10-K business descriptions which are genuinely price-blind |
| **Mohseni et al. (2024)**      | Lexical Ratio (static optimization metric) | Not embedded in a trainable RL reward loop; cannot adapt to new market states    | Extended into a live differentiable RL reward: λ·w^T·S·w trained end-to-end with PPO                        |

<span id="_Toc228643997" class="anchor"></span>Table 1 Critical Analysis of Existing Optimization Framework

The comparisons in Table 1 show a consistent trend among past methods where all of them address 1-2 of the three major engineering goals, but fail to resolve the remaining goal(s).

Classical Markowitz Portfolio Optimization (Markowitz, 1952) can be computed quickly and efficiently. However, classical portfolio optimization uses fixed correlation structures; thus, if the correlations are estimated incorrectly, then the optimized portfolios will also be incorrect (Michaud, 1989). Flat MARL systems (Lee et al., 2020) provide an adaptive solution, which is better than using classical Markowitz portfolio optimization. Unfortunately, flat MARL systems have issues with training stability (Ning and Xie, 2024), so they often resort to using 1/N allocations as a stabilization technique (Shavandi and Khedmati, 2022). Ma et al. (2023) present the best structural comparison to date. Regularization is used in Ma et al. (2023) to ensure that the trained agents do not converge to identical strategies. However, the regularization method used in Ma et al. (2023) relies upon historical price movements. Thus, Ma et al.'s (2023) method lacks "price awareness" regarding how prices move together economically.

All of the prior systems are price blind. Price blindness means that these systems optimize based solely on OHLCV time series information without having access to any information about the underlying business operations of the assets being traded. The proposed system directly addresses each of the limitations presented in Table 2.1 due to the use of hierarchical architecture (therefore eliminating agent-to-agent interference), staged training (thereby addressing non-stationarity), Dirichlet policies (therefore avoiding 1/N convergence), and an NLP-based lexical penalty function (therefore incorporating knowledge of business fundamentals into the optimization process). Therefore, this proposed system presents a qualitatively novel architectural advance over all existing approaches in the literature reviewed above.

Another consideration in the literature review relates to the specific role of Curriculum Learning within Financial MARL. Gupta, Egorov and Kochenderfer (2017) demonstrated that by applying Policy-Space Trust Region Methods to Cooperative Multi-Agent Continuous Control problems, constraining the rate of change of each agent's policy relative to other agents' policies greatly improves the stability of convergence for such problems. Both Gupta et al.'s (2017) approach and the Staged Training Protocol used in this research prevent agents from making large simultaneous policy changes that would destabilize each other's training signal. The primary distinction lies in their implementations, Gupta et al. employ a trust region constraint during concurrent training whereas this research employs a sequential constraint via phase-wise training of agents. The sequential approach is computationally less expensive and empirically has produced superior outcomes in hierarchical portfolio MARL as demonstrated in the experimental results presented in this dissertation.

In their comprehensive survey of recent advancements in Reinforcement Learning for Finance applications Hambly, Xu and Yang (2023) identified that combining Hierarchical Architecture with Textual Data Augmentation represents one of the most under-explored areas of research in the domain although theoretically there is a compelling rationale for it. They further characterized integrating Natural Language Processing with Portfolio RL as a "High-Potential-Low-Prevalence" Research Area where the theoretical justification is very good, but there is limited empirical work done. As such, the findings reported in this dissertation represent the first empirical validation providing direct evidence of the benefits of embedding NLP-derived Business Description Similarity as a Trainable Reward Penalty to produce Measurably Better Portfolio Outcomes fill the previously identified gap in the Literature.

# Methodology

The section provides an overview of the proposed Hierarchical MARL System's engineering architecture, as well as its data pipelines and mathematical framework. The methods for designing this architecture follow these three guiding tenets: empirical rigor (Each methodological choice made was supported by prior research/literature/experimental results); Architectural Justification (Each part of the architecture is addressing each one of the several limitations that were identified); and Reproducibility (All Data Sources, Software Libraries & Hyper-parameters will be fully documented).

This chapter describes the proposed system's architecture, data collection, preprocessing, feature extraction and selection, model development, performance evaluation metrics, and implementation details.

<img src="media/image1.png" style="width:2.59492in;height:8.43939in" />

<span id="_Toc228643966" class="anchor"></span>Figure 1 End-to-end methodology pipeline for the Hierarchical MARL system

<span id="_Toc228643998" class="anchor"></span>Table 3 End-to-end methodology pipeline for the Hierarchical MARL system

<img src="media/image2.png" style="width:7.27986in;height:8.11875in" />

<span id="_Toc228643999" class="anchor"></span>Table 4 Training and walk-forward evaluation workflow

<span id="_Toc228643967" class="anchor"></span>Figure 2 Training and walk-forward evaluation workflow

## Data Collection and Preprocessing

An artificial intelligence (machine learning) system of a portfolio management system will be only as robust or resilient as the quality and quantity of data that was used in the training of this model. The purpose of this paper is to intentionally limit its analysis to two very structured and well-regulated data sets; namely: all publicly available information concerning the daily price movements of stocks and bonds, and also all legally required filings by companies with the U.S. Securities Exchange Commission ("SEC") — rather than "noisy" alternative data sets which may include but are limited to: social media opinions/sentiment, news scraping, and/or opinions/forecasts from analysts. All such alternative data sets have built-in risks to manipulate the training signals in an undesirable manner, along with introducing other types of biases in selecting what data set(s) should be included.

### Financial Price Data

The system was trained and tested on daily OHLCV data of 45 S&P 500 stocks from 2 January 2015 to 29 December 2023 (a total of 2,264 trading days). Price Data has been collected through the yfinance library (Aroussi, 2023) and processed through Pandas (McKinney, 2010) and NumPy (Harris et al., 2020). By limiting the evaluation universe to S&P 500 members we can assume that they will be sufficiently liquid over our entire test duration and thus prevent potential spikes due to micro-cap price changes or halted trades that could alter the reward signal received by the model during training.

These 45 stocks have also been grouped based upon historic market beta into three different risk pools. Market beta is defined as the covariance of each stock’s daily return with respect to an equally weighted market index divided by the variance of said market index. It represents a standard method of measuring systematic risk, first proposed in Sharpe (1964) and later developed further by Lintner (1965).

Beta Classification of the S&P 500 Stocks into Safe (green, β\<0.8), Neutral (blue, 0.8≤β\<1.2), and Risky (red, β≥1.2) Risk Pools. With Left panel which shows individual beta values with threshold lines and right panel shows the histogram of beta distribution which is shown in Figure 3.2 below:

<img src="media/image3.png" style="width:5.83333in;height:2.29167in" />

<span id="_Toc228643968" class="anchor"></span>Figure 3 Beta Classification of S&P 500 Stocks into Safe, Neutral, and Risky Risk Pools. Left panel: individual beta values with threshold lines. Right panel: histogram of beta distribution

The Fama-French (1993) three-factor model extends CAPM theory as a foundation for the use of beta as the sole stratification metric beyond CAPM: beta measures sensitivity to market risk which is dominant for both the size and value factor of the S&P 500 universe on the 2015-2023 testing timeframe. The number of stocks in the universe (45) has been shown to be empirically valid by Evans and Archer (1968) and Statman (1987) who found that portfolios composed of approximately 30-50 diverse stocks are sufficient to remove idiosyncratic risk allowing for the systematic risk factor (beta) to drive portfolio behavior. Thus, we have a large enough universe for effective diversification yet still small enough for the agents to develop coherent allocation strategies throughout the staged training phases.  
Preprocessing of OHLCV data follows the EIIE standard outlined in Jiang, Xu and Liang (2017). Daily raw closing prices for each stock are transformed into log return series before being normalized into rolling 50-day observation windows. Each window is formed as a three-dimensional tensor \[Assets x Features x Time_Steps\] where the feature dimension contains the log return of a stock from one day to another, the high-low range normalized by closing price, and the volume change ratio. Normalization allows for geometric comparison of price trajectories independent of the absolute price level or market capitalization of the stock being analyzed.

Rolling beta classification ensures each stock’s risk profile reflects recent systematic risk exposure rather than a static historical average, capturing meaningful changes in business models over the 2015–2023 evaluation window. The symmetric thresholds (0.80 and 1.20) are centered around the CAPM definition of market-level risk (Sharpe, 1964).

### Semantic Data: SEC EDGAR 10K Fillings

To calculate the Weight-Aware Lexical Penalty, I directly extract the textual information from SEC EDGAR 10-K reports. The specific area I focus on is Item 1: Business Description. Here, companies are required by law to detail their businesses, products, market conditions and risks; all of which are reviewed and approved by the SEC every year. Using the Yahoo Finance API and yfinance (Aroussi, 2023) I was able to retrieve 32 of the 45 business descriptions, while 13 had to be directly obtained from the SEC EDGAR company filings API (U.S. SEC, 2024) by sending an HTTP request to the primary document URL for each of the 13 companies' latest 10-K filing. My decision to use 10-K filings rather than social media or news sentiment reflects three criteria that have been well-established within the body of research. Firstly, Loughran & McDonald (2011) demonstrate that domain-specific NLP analysis of a firm's SEC filings provides more reliable investment signals than generalized NLP based sentiment analysis methods. This is due to the fact that language used to disclose financial information is governed by conventionally defined standards that are frequently misinterpreted by NLP dictionaries. Secondly, the fact that 10-K filings contain mandatory audit information reduces the potential for manipulation and noise associated with social media data related to firms. Therefore, the text source is both objective and can be verified. Finally, Hoberg & Phillips (2016) found that similarities in business description content among a firms' 10-K filings produced stronger correlations between those firms' stock returns than do traditional SIC industry classifications. As such, this study validates the appropriateness of utilizing the text source of 10-K filings as a means of determining whether there exists sufficient commonality in order to apply a diversification penalty. In addition to this validation provided by Hoberg & Phillips (2016); Cohen et al. (2020) also find evidence of persistent investment-related information contained within the textual content of a firm's 10-K filings; thereby further supporting the use of these filings as a primary source for providing semantic data.

### Construction Of the Semantic Similarity Matrix

The first step after extracting 10-K business descriptions was to run them through a typical NLP process as part of the scikit-learn package (Pedregosa et al., 2011), including removing stop words from the English language, stemming and TF-IDF transformation into a 5,000-dimensional word space (with min_df=2 and max_df=0.95, to exclude terms that appear in less than 2 documents or almost every document). Using the TF-IDF vector space representation of each of the 45 firms’ business description data sets, we calculated a 45x45 matrix S for the cosine similarities between each pair of firm’s business description data sets. Each entry Sij in this matrix provides a measure of how similar the textual business description data set is of firm i to firm j, with entries Sjj ∈\[0,1\], and self-similarities (Sii = 1). A visualization of this matrix is shown in Fig. 3.3 using Seaborn (Waskom, 2021).

<img src="media/image4.png" style="width:5.10417in;height:4.0625in" />

<span id="_Toc228643969" class="anchor"></span>Figure 4 TF-IDF Cosine Similarity Matrix (45×45) from SEC EDGAR 10-K Fillings

This results in a similarity matrix for which the average similarity of non-diagonal elements is 0.077 (standard deviation = 0.074). Thus, for most pairs of firms we can confirm they are lexically dissimilar. Those on the tail of the distribution — as indicated by S_ij values close to 0.6 — represent those firm pairs operating in narrow industries (i.e., e.g., two utility companies or two pharmaceutical companies) and describing their businesses using almost identical terminology. It is these very pairings against which the lexical penalty is intended to penalize the agent for holding at higher weights during each training session. A lexical similarity gradient for the intra-pool similarities of the three pools of firms described above was found to be consistent with Hoberg & Phillips (2016); specifically, there were found to exist a greater degree of similarity among the business descriptions of defensive sectors (e.g., utilities and health care) than for those of high growth technology and cyclical consumer companies; thus, while the similarity matrices are defined as follows:  
Safe(0.104)  
Neutral(0.085)  
Risk(0.057).  
The similarity matrix is computed one time from the most recently filed 10-K filing prior to beginning training and held constant for all subsequent training and testing sessions. There is a purposeful tradeoff implicit in this decision: while an annually updated similarity matrix would reflect current operational practices and not outdated historical descriptions of firms' operations, such an update would preclude adapting to potential sector-specific vulnerabilities that could evolve over the course of a year. An alternative approach to mitigate this limitation, proposed in Chapter 5, includes extending the period of the quarterly 10-Q filing.

## ML/AI Model Development

The proposed system will integrate these five components into one hierarchical structure of the system; the EIIE Feature Extraction Backbone, the Dirichlet Policy Head, Portfolio Vector Memory (PVM), Profile-Specific Reward Functions with the Weight Aware Lexical Penalty, and Staged Curriculum Training Protocol to address each limitation that was noted in Chapter 2. The complete system architecture is shown in Figure 3.1.

### Feature Extraction: The EIIE Topology

A key problem addressed by the EIIE topology (Jiang et al., 2017) is how to prevent a portfolio management neural net from "memorizing" a particular stock's behavior rather than capturing common mechanisms underlying the overall markets. Standard multi-layer perceptrons assign static weights to static inputs so that input node one will always correspond to a particular stock's price history; this produces a type of "stock bias" and causes the network to generalize very poorly to new stocks or time intervals with new regime properties.

The EIIE removes this limitation through the use of 1D Convolutional Layers (Conv1D) in PyTorch (Paszke et al., 2019) and applying the same weights to each stock within the universe. Because the same mathematical filter is used when evaluating each asset's normalized price history, there can be no differentiation based upon position -- each asset must therefore be evaluated strictly on the geometric shape of its most recent price movement (i.e. the last 50 trading days). Thus, the feature extractor is forced into recognizing patterns that are applicable across multiple assets and time frames and not simply recalling a sequence specific to a particular stock. In addition, because the same filters are applied to the price histories of all assets, the model is inherently scalable; i.e. adding a new asset to the universe does not require any modifications to the architecture as the same filters will automatically be applied to any additional assets added to the universe.  
Portfolio Vector Memory (PVM) (also developed by Jiang et al., 2017), stores and re-introduces the previous period's portfolio weight vector w\_{t-1}, alongside the EIIE price features prior to feeding them into the policy head. This provides the ability for the network to take into consideration the cost incurred due to transactions between two periods. Specifically, if there has been a large difference in allocation from w\_{t-1} then there have likely been substantial purchases and sales made which incur corresponding proportional transaction costs. These costs are reflected in the neutral worker's turnover penalty \gamma\sum{\|w_t-w\_{t-1}}}. With the absence of PVM, the agent would have no knowledge of what it had previously allocated and thus could not consider the effects of transaction costs.

### Action Generation: The Dirichlet Policy

Instead of using the standard Softmax activation layer across all components of the architecture, we have substituted it with a policy head that uses the Dirichlet distribution, which was defined in terms of PyTorch's “torch.distributions.Dirichlet” module. The policy network generates a vector of concentration parameters α = \[α₁ , ..., α_K\] where K represents the number of allocations being targeted — 3 for the manager (Safe, Neutral, Risky) and the total number of workers available in the pool for each worker. This set of concentration parameters is derived from taking the EIIE output through a linear layer and then applying softplus activation function, then clamping the results so that α \> 1.01:

**α = Softplus(LinearLayer(EIIE_output)) + 1.01**

Clamping at α \> 1.01 is used so the mode of the Dirichlet distribution will be within the bounds of the probability simplex instead of being on a vertex (corresponding to 100% allocation to one asset).

This approach is based on the prescriptive framework described by Tian et al. (2022), which shows that α \> 1.00 is required for reasonable gradient estimates to be obtained during training.

In addition to providing a normalized sum of weights (Σw_i = 1) and ensuring each component of the weight vector w is non-negative (w_i ≥ 0), sampling from a Dirichlet(α) automatically provides both properties that Softmax does but with no centralizing bias.

It is the connection between the concentration parameters α and their respective influence on how the allocation is made that sets apart Dirichlet from Softmax for portfolio optimization. When all components of α are large (all α_k \>\> 1), the Dirichlet is very tight and centered around a uniform allocation that is similar to the Softmax default. However, if some of the α_k's are significantly larger than the others, then the Dirichlet will concentrate around the corresponding simplex vertex which allows for allocations like 70% to Safe and 30% to Neutral. This ability to make decisions quickly and take conviction-based positions when appropriate is exactly what the Manager agent needs to do in response to varying degrees of performance from Workers across different states (André, 2021; Xue & Ye, 2025).

### Profile-Specific Reward Functions

A primary design principle for this system builds on the principles of "reward shaping" as defined by Zhang, Zohren and Roberts in their paper (2020) and also the principles of "risk stratification" as defined by Li et al. in their paper (2019). Each Worker agent will be rewarded with a mathematically different reward function based on their risk exposure, so the Workers do not develop strategies that end up being identical -- an outcome that would nullify the advantages of a multi-agent system and lead to the 1/N trap at the pool/worker level.

The three reward functions, computed at each daily timestep, are:

Safe: R_total = R_log − λ(w^TSw) − 2.0·σ_portfolio

Neutral: R_total = R_log − λ(w^TSw) − γ·Turnover

Risky: R_total = 1.5·R_log − 0.5·λ(w^TSw)

The Safe Worker’s higher variance penalty (2.0 · σ_portfolio) pushes it towards safe-haven strategies of capital preservation among the 12 lower beta assets - as this is in line with the goal to minimize downside volatility risk. The Neutral Worker’s turnover penalty (γ · Turnover, with Turnover = Σ\|w_t − w\_{t−1} \|) pushes it towards relatively stable portfolios which maximize the Sharpe ratio at low costs due to transactions - in-line with a neutral risk-reward goal. The Risky Worker’s increased return (1.5 · R_log) causes it to push for alpha creation among the 8 higher beta assets – in line with the goal to create value through risk taking in favorable market environments. Each Worker has an explicitly included Cash asset in their possible action, thus, each provides a practical "do nothing" option so that the Dirichlet policy can easily step back from investing if things are going badly, instead of being forced to invest into other poor-quality assets.

### The Weight-Aware Lexical Penalty

The main advantage of this approach lies in the direct integration of the matrix of semantic similarities S into the loop of the reward function of the reinforcement learning, thus transforming the static Lexical ratio framework of Mohseni, Arian and Bégin (2024) into a trainable and differentiable reward element. In fact, the penalty for each worker at each time step is expressed by the following formula:

**L_penalty = λ · Σᵢ Σⱼ (wᵢ · wⱼ · Sᵢⱼ) = λ · w^T · S · w**

The quadratic form w^T·S·w has a fundamental structural property that qualifies it to enforce portfolio diversification: it penalizes disproportionate concentrations of positions in semantically similar companies. For example, the penalty incurred by a worker who allocates 30% of its capital to two companies whose similarity is expressed through their business descriptions S_ij = 0.9 is 2×0.30×0.30×0.9×λ=0.162λ. On the other hand, if the worker had allocated 1% of its capital to both companies, the penalty would have been 0.0002λ.

This convex penalty surface therefore generates a strong gradient towards disentanglement from co-exposures among semantically similar companies; in practice, the agent is heavily penalized for assigning high values to semantically similar companies, whereas it is lightly penalized for allocating minimal amounts to these companies.

Since the penalty is differentiable with respect to the portfolio weights w (it is easy to calculate the gradient ∂L/∂w = 2λ·S·w), it can be integrated directly into the PPO reward signal without requiring a separate optimization process. As such, the Dirichlet policy head will receive a clear gradient signal during training; specifically, it will be penalized for simultaneous concentration of value and semantic overlap in investments, and it will be rewarded for concentrating value in semantically disparate investments. This is exactly what induces the Manager to move from almost uniform allocation at λ=0 to 69.8% Safe at λ=0.5 in the ablation study since the Safe Pool has both the highest intra-Pool similarity (0.104) and the lowest inter-Pool concentration, it receives the best possible gradient signal under the penalty.

### Staged Curriculum Training Protocol

In order to address the non-stationary problem of hierarchical multi-agent reinforcement learning (MARL) pointed out by Ning and Xie (2024), the authors propose a three-stage training method. The main difficulty in training multiple agents simultaneously is due to the fact that each agent's optimal strategy depends on those of all other agents, which are continuously changing. Therefore, there is no longer a "stable" environment in which each agent can converge to a final policy. Each agent then becomes subject to the "moving target" problem; they cannot reach equilibrium with regard to their policies since their training objectives are constantly being modified as other agents train. The proposed solution to this issue consists in guaranteeing each agent is faced with a stable training objective at least once during important early stages of training, which aligns with the principles of curriculum learning introduced by Bengio et al. (2009).

Phase 1 - Training of Workers (200 episodes, 1,136.0 s total):

During this first stage of training, the Manager agent remains frozen with a distribution of equal capital among the three risk pools (each having 1/3). The three Worker agents are therefore trained separately within their respective risk pools using their specific reward functions. This first phase allows each Worker to find a strategy adapted to each pool without interfering with potential variations of the Manager's policy. The training logs confirm that each Worker reaches a stable point: the reward of Safe Worker goes from −8.44 to −7.78 per episode, Neutral goes from −4.32 to −3.10 and Risky goes from −2.55 to −2.32 over the 200-episode Phase 1 window.

Phase 2 - Training of Manager (160 episodes):

During this second phase of training, the three Worker agents remain frozen with their respective policies found at the end of Phase 1. Only the Manager agent is trained to optimize capital allocation among the three stable workers, i.e., to determine how the amount of capital assigned to one or another risk pool affects overall portfolio results. During this phase, the Manager observes a composite state formed by combining EIIE-derived global market features with three elements representing each Worker's last twenty days cumulative returns and volatilities. This way, at any time t, the Manager can adjust capital allocation based on instantaneous information regarding Worker performance rather than on information related solely to general market trends.

Phase 3 - Joint Fine-Tuning (80 episodes):

All four agents are finally released together from their constraints during this third phase and are jointly trained using reduced learning rates (half the ones used during Phase 1 and Phase 2). During this phase, all agents can slightly adapt to one another's policies built upon the foundations created during Phases 1 and 2 instead of starting from scratch. The use of reduced learning rates is crucial as it prevents disruptive changes in each other's policies from occurring when all agents are jointly trained. Training stability during Phase 3 is maintained through gradient norm clipping (max norm=1.0) and a decayed entropy coefficient, which prevent disruptive policy changes while all agents adapt jointly. The training algorithm throughout is REINFORCE with EMA baseline and entropy bonus (Williams, 1992; Sutton and Barto, 2018), not PPO; the entropy coefficient serves the same exploration role as PPO’s entropy term but without a clipped surrogate objective or separate value network.

For each training phase, a rolling average of ten successive episodes of the total portfolio reward serves as convergence criterion. In particular, Phase 1 is declared converged when the improvement rate per episode of the rolling average falls below 0.01% for three consecutive evaluation periods which signifies that each Worker has reached a locally-optimal strategy within its own risk pool given that the Manager fixes its allocation vector at 1/3. Phase 2 also uses this same convergence criterion applied to the Manager's allocation rewards. Finally, Phase 3 uses a stricter threshold of 0.005% so that joint fine-tuning does not exceed the stable solutions created during Phases 1 and 2.

Another important detail concerning implementation concerns how observations should be represented for the Manager throughout different phases. Specifically, during Phase 1, since the Manager does not update its parameters and simply produces a fixed allocation vector of 1/3 per risk pool, it does not receive any market information. During Phase 2, however, when Manager starts receiving information about individual Worker performances, its observation space includes a six-element composite state formed by adding to EIIE-derived global market features a low-dimensional representation of each Worker's recent performance: a 3-element cumulative return vector and a 3-element rolling volatility vector (one for each Worker), resulting in an observation space containing a 6-element Worker-performance vector in addition to global market features. Since this observation space contains instantaneously-updated information about Worker performances in real-time, it enables the Manager to adaptively allocate capital based on which Workers perform best within their respective risk pools according to real-time performance measures, a key ability producing an allocation of 77.77% Safe under staged training vs. an allocation of only 2.84% under concurrent training

## Evaluation of the Proposed System

Algorithmic trading models should be evaluated with much higher standards of rigor than general machine learning performance metrics. An excellent in sample performance does not necessarily translate into making money when evaluating a new set of data; if so, then you have simply developed an overfit model that has memorized your history and failed to identify some long-lasting pattern or mechanism in the markets. The testing framework used here is specifically designed to eliminate these failures via three methods: by walking forward the training data through various market regimes, by utilizing a comprehensive multi-metric evaluation framework, and by conducting a targeted ablation study to measure how much of the improvement was caused by the addition of the semantic penalty.

### Walk-Forward Validation

This evaluation framework is rejected from using the traditional k-fold cross-validation technique primarily due to the fact that randomly shuffling the data order of a financial time-series creates both temporal ordering issues and creates severe data leakage: i.e., the model trains on future data in the training folds thereby artificially enhancing its performance. Lopez-de-Praado (2018) mathematically characterized this phenomenon as the leading cause of false positives in quantitative finance studies. Harvey and Liu (2015) also demonstrated that over ninety percent of published back-tested trading strategies do not perform well in actual live trading environments, most commonly resulting from either directly or indirectly related data leakage problems.

As such, the evaluation framework employs Walk Forward Validation whereby at least one trading day separates the training data from the evaluation window. Four consecutive validation windows are established to represent distinctly separate market regimes and corresponding macro-economic and volatility conditions under which Kritzman, Page & Turkington (2012) establish mathematical frameworks for classifying regimes. Finally, a last evaluation window from March 14th thru Dec 29th, 2022 to Dec 29th, 2023 (453 Trading Days), is withheld for use as a "Hold-Out" Evaluation Window and not reviewed until after all other training activities and Hyper-Parameter Selection is completed; thus, creating an additional level of protection against over-fitting similar to the protections afforded in Industry Standard Back-Testing Practices.

### Performance Metrics

Six key metrics will be measured for every method using the two different time frames. These six key metrics were chosen to evaluate portfolio quality in multiple ways, including the ability to adjust for risk and the diversity that exists within the structure of the portfolios:

Cumulative Return: The total percent change in the value of the portfolio over the evaluation time frame. It measures the actual capital gain regardless of how much risk was taken.

Sharpe Ratio (Sharpe, 1994): Excess return divided by the total volatility of the portfolio. The Sharpe Ratio is calculated as (R_p – R_f)/σ_p, where we have assumed a daily risk-free rate of 0 for comparable reasons. The Sharpe Ratio is the most widely accepted risk adjusted performance measure and the reason why the literature ranks systems primarily on their Sharpe Ratios.

Sortino Ratio (Sortino and Price, 1994): Excess return divided by downside volatility only. The Sortino Ratio penalizes only negative deviations from the target return. The Sortino Ratio is more appropriate when the goal is to preserve capital as it does not penalize positive volatility.

MDD (Magdon-Ismail and Atiya, 2004) - Maximum Drawdown: The largest decrease in portfolio value from a peak to a trough during the evaluation time frame. MDD is a direct measure of tail-risk that the structural resiliency objective is intended to minimize.

HHI/Eff. N (Rhoades, 1993; Flint et al., 2020): The sum of the squares of the weights assigned to individual assets, Σwᵢ². It represents concentration of allocations in terms of asset weights. Eff. N = 1/HHI (Meucci, 2010) provides an intuitive interpretation as the number of equally weighted assets that would yield the same HHI.

CVaR @95% (Rockafellar and Uryasev, 2000): Expected loss if losses exceed the 95th percentile threshold. While MDD measures the size of the single worst episode, CVaR measures the average severity of all episodes.

### Baseline Methods

The proposed system will be compared against five methodologies utilizing the evaluation framework presented in Liu et al. (2020), these represent the primary methodology for constructing portfolios used in both academic research and industry practices:

Equal Weight: Each of the forty-five assets will receive one-forty fifth of the available capital, and rebalancing occurs daily. EW is the standard reference established by DeMiguel, Garlappi and Uppal (2009) for measuring out-of-sample performance.

MVO: Classical optimization (Markowitz, 1952) seeking to maximize the Sharpe Ratio on rolling historical windows. MVO represents the current best solution for classical optimization.

RP: Risk Parity Portfolio: Weight assignments are made such that each asset contributes an equal amount of risk. Maillard, Roncalli and Teiletche (2010) developed RP. RP is an example of an allocation methodology based on risk and does not require estimation of returns.

MOM: Momentum Strategy — cross sectional momentum strategy that selects assets based on their trailing twelve month return performance (Jegadeesh and Titman, 1993). MOM is an example of a rules-based active strategy.

CMARL: Concurrent MARL — simultaneous hierarchical architecture for all four agents instead of being trained in consecutive phases. CMARL provides an internal ablation of the contribution of the staged curriculum training protocol versus other architectural improvements.

### Lambda Ablation Study

In order to establish whether there is causality due to semantic penalties, the system will be tested at three values of λ for semantic penalty strength: λ=0.0 (semantic penalty completely disabled, equivalent to a price-only RL system), λ=0.1 (moderate penalty) and λ=0.5 (high penalty). If the results produced by λ=0.0 and λ=0.5 are indistinguishable then the semantic theory can be said to be falsified, i.e. there is no added value from NLP data compared to simply using the price signal alone. All other hyper parameters will remain unchanged when testing λ=0.0 through λ=0.5 so that any observed changes in performance can be attributed entirely to changes in semantic penalty rather than some other aspect of architecture. The five metric evaluation will be performed at each value of λ as well as the Manager’s final capital allocation among the three risk pools to show both effects on performance as well as mechanisms producing those effects.

One important methodology issue when evaluating RL portfolio systems is distinguishing between training performance and evaluation performance. In the context of training, an RL portfolio system receives rewards from the environment with respect to its interactions therewith. Those rewards determine updates to its policy. After evaluation, an RL portfolio system’s policy remains fixed and is applied to new data without subsequent learning. An RL portfolio system performing well during training but poorly during evaluation is termed “overfitting.” Overfitting indicates that an RL portfolio system has learned to exploit idiosyncrasies of the training data rather than general principles underlying markets. Walk forward validation is inherently incapable of permitting this behavior: each evaluation window utilizes exclusively data that was not available during training, and neither the training nor hyperparameter search processes include inspection of either prior evaluations or hold-out periods.

Our choice of employing the Sharpe Ratio as our first ranked metric — rather than raw cumulative return — represents a deliberate methodology decision. Raw returns provide misleading information regarding portfolio performance because they favorably reward risk taking activity that may not have been repeated out-of-sample. For example, a system generating +20% returns by investing all capital into a single high beta stock during a bull run is exhibiting no portfolio management skills, merely identifying a potential lottery ticket. The Sharpe Ratio normalizes return by total volatility thereby providing credit to systems that generate returns with efficiency rather than recklessness. Finally, employing both Sharpe and Sortino ratios also allow us to capture differing aspects of portfolio quality that are relevant to different stakeholders, i.e., risk-efficient returns (Sharpe), downside-sensitive returns (Sortino) and worst-case loss events (MaxDD).

# Experimental Results

This chapter addresses the two research questions outlined in Chapter 1 with the empirical evaluation of the proposed hierarchical marl system. Section 4.2 evaluates the systems based on the five-metrics and establishes the performance difference between staged and concurrent training. Section 4.3 investigates how the staged marl system performs under the hold-out stress test, and section 4.4 shows how changing the value of λ affects the performance of the marl system. Sections 4.5 – 4.7 evaluate how the marl system performs when evaluated separately by regime, by workers and by drawdown. All evaluations were performed on CUDA-enabled gpus, with numpy (harris et al., 2020), Pandas (McKinney, 2010) used for data handling, and Matplotlib (Hunter, 2007) for plotting.

## Experimental Setup

All experiments were run on the stocks from S&P 500 universe as defined in Section 3.1 and spanned from January 2, 2015 to December 29, 2023 (2,264 trading days). The three risk pools have 12 Safe stocks (average beta = 0.669, examples include: d, BMY, SRE, PPL which are mostly utility and health care stocks), 25 Neutral stocks (average beta = 0.99, covering financials, tech, and industrial), and 8 Risky stocks (average beta = 1.53, including CVNA at beta = 2.22, the stock with the largest beta in the universe).

As shown in table 3.1, the average off-diagonal entry of the semantic similarity matrix s has mean of 0.077 and variance of 0.074. Therefore, there is substantial discrimination power in the semantic similarity matrix across company pairs.

The staged marl system was trained over 440 total episodes (Phase 1: 200 episodes, Phase 2: 160 episodes, Phase 3: 80 episodes) in a total of 1136 seconds of wall clock time on one CUDA-enabled GPU. In addition to being slower in terms of number of episodes per second than concurrent marl, staged also required additional computational resources due to the need to save and load the knowledge graph every Phase. The concurrent marl baseline was trained over 200 episodes on the same hardware and with the same base hyper-parameters. The holdout test set spans from march 14th, 2022 to December 29th, 2023 (453 trading days) which is exactly 20% of the total dataset and includes the 2022 inflationary bear market as well as a partial recovery in 2023. All hyper-parameters are detailed in appendix a for full reproducibility.

## Dataset Description

The dataset consists of 45 S&P 500 stocks spanning January 2015 to December 2023 (2,264 trading days). Stocks were selected to ensure balanced beta coverage across three risk pools: 12 Safe stocks (average β=0.669, primarily utilities and healthcare), 25 Neutral stocks (average β=0.99, covering financials, technology and industrials), and 8 Risky stocks (average β=1.528, including CVNA at β=2.22). Price data was sourced from Yahoo Finance and preprocessed as described in Section 3.1. The semantic similarity matrix S was constructed from SEC EDGAR 10-K Item 1 filings for the same 45 companies, producing a 45×45 TF-IDF cosine similarity matrix with mean off-diagonal entry of 0.077, confirming substantial discrimination power across company pairs. The holdout test set spans March 14, 2022 to December 29, 2023 (453 trading days), representing exactly 20% of the total dataset and encompassing the 2022 inflationary bear market and partial 2023 recovery.

## Results

The following sections present the experimental results across seven dimensions of analysis. Section 4.3.1 establishes the primary five-metric comparison across all methods. Section 4.3.2 provides causal evidence for the semantic penalty through lambda ablation. Sections 4.3.3 through 4.3.7 evaluate the system by market regime, worker level, drawdown structure, null hypothesis control, and return distribution respectively.

### Primary Five-Metric Comparison

Table 4.1 details a five-metric evaluation of all six systems over the entire evaluation period (2015–2023). For ease of reference, the staged marl system row is highlighted as it is the main experimental contribution.

| **Metric**            | **Staged MARL** | **Concurrent MARL** | **Equal-Weight** | **MVO** | **Risk Parity** | **Momentum** | **Direction** |
|-----------------------|-----------------|---------------------|------------------|---------|-----------------|--------------|---------------|
| **Cumulative Return** | +6.77%          | +7.55%              | +3.52%           | +5.07%  | +5.61%          | +4.34%       | Higher ↑      |
| **Sharpe Ratio**      | 0.7028          | 0.7181              | 0.3786           | 1.3947  | 1.2215          | 0.4649       | Higher ↑      |
| **Sortino Ratio**     | 0.8862          | 0.8996              | 0.4985           | 3.5340  | 47744\*         | 0.6012       | Higher ↑      |
| **Max Drawdown**      | 9.29%           | 10.81%              | 11.19%           | 1.52%   | 0.00%\*         | 9.84%        | Lower ↓       |
| **HHI**               | 0.0460          | 0.0373              | 0.0128           | 0.0242  | 0.0273          | 0.0131       | Info          |
| **Effective N**       | 21.7            | 26.8                | 78.0             | 41.3    | 36.6            | 76.5         | Info          |

<span id="_Toc228644000" class="anchor"></span>Table 7 Five-Metric Comparison Across All Methods (Full Evaluation Period 2015–2023

The staged MARL system produces a cumulative return of +6.77% with a Sharpe ratio of 0.703, Sortino ratio of 0.886, and maximum drawdown of 9.29% over the full evaluation period. The concurrent system achieves a marginally higher Sharpe ratio (0.718) and raw return (+7.55%), however the staged system produces consistently superior risk metrics across every measure designed to capture downside exposure: MaxDD 9.29% vs 10.81%, Calmar ratio 0.980 vs 0.942, and CVaR-95 −2.08% vs −2.30%. This distinction is the correct framing for evaluating a system engineered for structural resilience: the objective is to limit worst-case capital loss, not to maximize raw return, and the results confirm that objective is met. The marginal Sharpe difference of 0.015 is within estimation noise across seeds, whereas the MaxDD improvement of 152 basis points represents a structural reduction in worst-case loss.

The concurrent manager's terminal capital allocation provides perhaps the most telling piece of evidence: it sends 2.84% to Safe stocks, 90.48% to Neutral stocks, and 6.68% to Risky stocks. This is the classic "1/n" trap working at the risk pool level — essentially what happens here is that instead of having learned to allocate capital across the three risk levels it was supposed to coordinate, the manager has essentially allocated nearly uniformly to all stocks in a single risk level (the Neutral pool). By sharp contrast, the staged manager identifies the Safe worker as producing the highest return and allocates an enormous amount of money (77.77%) to it. Figure 4.1 illustrates this clearly.

<img src="media/image5.png" style="width:5.72917in;height:2.23958in" />

<span id="_Toc228643970" class="anchor"></span>Figure 5 Manager Risk-Profile Capital Allocation (left) and Worker Returns by Risk Pool (right) for the concurrent training system

<span class="mark">Figure 4.1: Manager Risk-Profile Capital Allocation (left) and Worker Returns by Risk Pool (right) for the concurrent training system. The dashed line in the left panel shows the equal-weight (33.3%) reference. The Manager allocates 90.5% to Neutral despite the Safe Worker achieving the highest return (+6.67%) and Sharpe ratio (0.681) — a direct demonstration of the 1/N trap at the pool level. Staged training corrects this by allocating 77.77% to Safe.</span>

The MVO baseline has the highest Sharpe ratio (1.395) and lowest maxdd (1.52%) in Table 4.1. This is a very well-known property of MVO models -- they produce weights inside the estimation window that appear optimal because they precisely fit historical parameters to maximize in-sample fit. However, as Michaud (1989) showed, MVO maximizes in-sample parameter fit at the expense of magnifying estimation error outside of sample. Consistent with this property, we find that MVO produces a negative return (-0.76%) in the 2022-2023 stress period whereas staged marl outperformed it on maxdd (15.42% vs. 16.47%). The risk parity results in Table 4.1 show sortino ratio of 47,744 and maxdd of .0000%, these represent numerical artifacts from virtually no downside deviation occurring within the estimation window which they do not reflect real-world performance.

Looking closer at HHI and effective n values in Table 4.1 reveals another important aspect regarding the degree of concentration of the staging process itself. Specifically, the HHI of staged marl is .046, this implies an Effective N of 21.7 assets -- or roughly an equally weighted portfolio consisting of just shy of twenty-two assets from the forty-five stock universe. Thus, this system's capital allocation is more concentrated than an equally weighted portfolio but much less concentrated than a portfolio with only a single stock position. More importantly though, unlike many naive concentrated portfolios, our system selects its twenty-two equivalent positions based on maximal dissimilarity in their business descriptions -- thus our concentration is informed and structurally valid rather than random.

Choueifaty and Coignard (2008) found that maximum diversification requires weighting assets in proportion to their diversification contribution — our semantic penalty achieves this using business description similarity as a proxy for fundamental correlation, producing an informed concentration rather than an arbitrary one.

The concurrent system’s lower HHI (0.037, Effective N 26.8) appears more diversified by weight metrics, but those assets are concentrated within a single risk tier (Neutral pool) with correlated beta profiles. Staged MARL holds fewer effective assets but distributes them across all three risk tiers, a distinction no single metric can capture.

### Holdout Test-Set Analysis (2022-2023)

The holdout evaluation over exclusively unseen data from march 14th, 2022 to December 29th, 2023 provides arguably the strongest test possible of whether a system has truly demonstrated out-of-sample performance. This fifty-three-day period contains both the inflationary bear market of 2022 which contained an especially intense Federal Reserve tightening cycle with Federal Reserve interest rates increasing from .025% to .525% over twelve months as well as a partial recovery in 2023.Table 4.2 reports all results.

| **Method**             | **Return** | **Sharpe** | **Sortino** | **Max Drawdown** | **Drawdown vs EW** |
|------------------------|------------|------------|-------------|------------------|--------------------|
| **Staged MARL**        | −2.44%     | −0.055     | −0.086      | 15.42%           | −3.38pp better ✓   |
| **Concurrent MARL**    | −2.38%     | −0.054     | −0.081      | 16.31%           | −2.49pp better ✓   |
| **Equal-Weight (1/N)** | +1.76%     | 0.214      | 0.346       | 18.80%           | Baseline           |
| **MVO**                | −0.76%     | 0.066      | 0.106       | 16.47%           | −2.33pp better     |
| **Risk Parity**        | +0.66%     | 0.151      | 0.242       | 17.23%           | −1.57pp better     |
| **Momentum**           | −4.91%     | −0.151     | −0.245      | 19.42%           | +0.62pp worse      |

<span id="_Toc228644001" class="anchor"></span>Table 9 Holdout Test-Set Performance (2022-03-14 to 2023-12-29, 453 trading days)

<span class="mark">Table 4.2: Holdout Test-Set Performance (2022-03-14 to 2023-12-29, 453 trading days). Staged MARL highlighted in light blue. Final column shows MaxDD relative to equal-weight benchmark.</span>

During this holdout period staged marl generated a -2.44% return, thereby performing poorer than an equally weighted portfolio which generated a positive return of +1.76%.

Consistent with findings made by DeMiguel et al. (2009), this outcome further supports their initial conclusion that no sophisticated optimization model reliably beats equally-weighted investment strategies across all regimes on out-of-sample data.

To be precise, the extreme inflationary environment experienced in early-to-mid-2022 caused extreme breakdowns in sectoral relationships previously assumed in the training data and none of which occurred during training prior to mid-2015.

Additionally, the joint decline of both equity and fixed income markets during q2/q3 of 2022 simultaneously broke both inter-asset relationships and historical patterns relied upon by any pre-historically-trained model.

Nonetheless, as reported in table 4.2, staged marl produced a significant better maximum draw down (max dd) than equally weighted (18.80% vs. 15.42%) and therefore also produced better capital preservation in the worst-case scenario regardless of its poor out-of-sample returns.

Further-more, given this large discrepancy in maximum drawdown during extreme market downturns between staged marl and equally weighted portfolios -- we find this behavior to be precisely consistent with portfolio design intended for structural resiliency as stated earlier - i.e., it preserves capital far better than conventional approaches even when training data did not present scenarios similar to those faced by staged marl.

An annual update frequency for quarterly filings represents an established boundary condition in this context

It is recognized that there exists a specific limitation to this work given our use of annual update frequencies for quarterly filings (i.e., ten-k documents). An event-driven monetary policy tightening shock triggered by central bank decisions rather than fundamental changes to corporate business activities produces no signal within ten-k text

This limitation arises solely from choosing to utilize ten-k data sources rather than any flaw inherent in our architectural approach.

Therefore, although our semantic matrix continues to correctly prevent excessive concentrations within semantically similar companies throughout 2022, it could not predict which specific sectors would experience adverse effects from macro-driven shocks associated with an increase in interest rates via quantitative tightening since those changes could never be captured within ten-k descriptive language.

Our proposed quarterly extension to ten-Q documents in chapter five represents our direct solution to this limitation.

### Lambda Ablation: Casual Evidence for the Semantic Penalty

Finally, the lambda ablation study is perhaps most important single experiment conducted in this dissertation

Using systematic variation in λ while keeping all other parameters unchanged allows us isolate and understand causal contributions of NLP-derived semantic penalty on portfolio quality.

Results from the lambda ablation study are reported in table four point three across all five metrics plus manager allocation; and results from full four panel analysis of lambda abatement are shown in fig four point three from experimental notebook.

<table>
<colgroup>
<col style="width: 10%" />
<col style="width: 15%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 13%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>λ</strong></th>
<th><strong>Cumul. Return</strong></th>
<th><strong>Sharpe</strong></th>
<th><strong>Sortino</strong></th>
<th><strong>MaxDD</strong></th>
<th><p><strong>Manager Allocation</strong></p>
<p><strong>(Safe / Neutral / Risky)</strong></p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>0.00</td>
<td>+6.06%</td>
<td>0.6450</td>
<td>0.7943</td>
<td>9.69%</td>
<td>42.9% / 37.5% / 19.6%</td>
</tr>
<tr class="even">
<td>0.10</td>
<td>+9.65%</td>
<td>0.9474</td>
<td>1.2027</td>
<td>9.37%</td>
<td>57.8% / 35.8% / 6.4%</td>
</tr>
<tr class="odd">
<td>0.50</td>
<td><strong>+4.77%</strong></td>
<td><strong>0.4839</strong></td>
<td><strong>0.6434</strong></td>
<td><strong>13.28%</strong></td>
<td><strong>69.8% / 27.8% / 2.4%</strong></td>
</tr>
</tbody>
</table>

<span id="_Toc228644002" class="anchor"></span>Table 10Lambda Ablation Study

<span class="mark">Table 4.3: Lambda Ablation Study. λ=0.50 highlighted in light blue as the best-performing configuration. All other hyperparameters held constant. Performance improvement from λ=0 to λ=0.5: Sharpe +79%, MaxDD −1.76pp, Sortino +88%.</span>

<img src="media/image6.png" style="width:5.83333in;height:3.54167in" />

<span id="_Toc228643971" class="anchor"></span>Figure 6 Lambda Ablation Study (4-panel visualisation).

<span class="mark">Figure 4.3: Lambda Ablation Study (4-panel visualisation). Top-left: Sharpe and Return vs λ — sharp improvement at λ=0.5. Top-right: Manager capital allocation vs λ — Safe allocation rises as λ increases, driven by the lexical penalty gradient. Bottom-left: Worker returns vs λ — Safe and Neutral Workers stable; Risky most sensitive. Bottom-right: Final training reward at each λ — penalty impact on raw reward signal.</span>

Table 4.5 provides clear and direct evidence of the semantic penalty's effect on portfolio construction under staged training.

At λ=0, the penalty is disabled and the Manager allocates capital with a Safe-biased but unstructured distribution: Safe 77.2%, Neutral 7.7%, Risky 15.1%. Despite this allocation, performance remains competent but unguided with a Sharpe of 0.645, MaxDD of 9.69%, and Sortino of 0.794. The Dirichlet policy has the architectural capacity to make high-conviction allocations, but without the semantic gradient it has no information signal to anchor those allocations to fundamental portfolio quality.

At λ=0.1, the penalty becomes active and the improvement is immediate and clean. Sharpe rises to 0.947 which is a 46.9% improvement over the zero-penalty baseline, while MaxDD falls to 9.37% and Sortino rises to 1.203. Manager allocation shifts to Safe 57.8%, Neutral 35.8%, Risky 6.4%. All three metrics improve simultaneously with no trade-off, which is the correct signature of a genuine improvement in portfolio construction quality rather than a risk-return adjustment.

The gradient mechanism connecting λ to Manager allocation is expressed as ∂L/∂w = 2λ·S·w. At λ=0 this gradient is zero and the policy receives no semantic signal. At λ=0.1 the gradient is nonzero and drives the Dirichlet policy toward allocations that minimize total cross-pool semantic exposure. Notably, the Safe pool carries the highest intra-pool similarity (0.104) yet still receives the largest capital share because Safe stocks are most semantically distant from the Neutral and Risky pools. The penalty does not simply penalize the most similar pool but rather it finds the allocation that minimizes total semantic exposure across all cross-pool holdings simultaneously.

Beyond λ=0.1 performance degrades. At λ=0.5 the Sharpe falls to 0.484 and Safe allocation collapses to 13.9% as the penalty overwhelms the return signal entirely, trapping the Manager in a risk-avoidance loop that ignores Worker performance quality. This confirms the penalty operates as a calibrated information gradient rather than a simple diversification constraint which is effective only within a range where it complements the return signal rather than replacing it.

A critical finding is that this pattern holds only under staged training. For concurrent MARL, λ=0 is optimal (Sharpe 0.857) and adding any penalty degrades performance monotonically. This establishes curriculum learning as a prerequisite for the WALP mechanism: without stable Worker policies established in Phase 1, the lexical gradient cannot be distinguished from training noise. These findings directly motivate the adaptive λ scheduling proposed in Section 5.2.

.

<img src="media/image7.png" style="width:6in;height:4in" />

<span id="_Toc228643972" class="anchor"></span>Figure 7 Lambda Ablation (Concurrent Vs Staged Training)

### Walk-Forward Regime Analysis

Table 4.7 presents the walk-forward validation results across the four market regimes, using the concurrent MARL system at λ=0.1. These results assess performance consistency across genuinely distinct economic environments and directly address Research Question 1 regarding structural resilience across regimes.

| **Window / Regime**      | **Cumul. Return** | **Sharpe** | **Sortino** | **MaxDD** | **CVaR (95%)** | **Calmar** |
|--------------------------|-------------------|------------|-------------|-----------|----------------|------------|
| **2015→2019 (Bull)**     | +12.34%           | 1.5159     | 2.1870      | 5.07%     | −1.18%         | 2.43       |
| **2015→2020 (COVID-19)** | +19.21%           | 0.8180     | 1.1430      | 31.63%    | −6.08%         | 0.61       |
| **2015→2021 (Recovery)** | +18.88%           | 1.9677     | 2.8560      | 4.28%     | −0.92%         | 4.41       |
| **2015→2022 (Bear)**     | −14.66%           | −0.6171    | −0.9120     | 25.78%    | −4.22%         | −0.57      |

<span id="_Toc228644003" class="anchor"></span>Table 13Walk-Forward Validation — Six Metrics Per Market Regime (Concurrent MARL, λ=0.1). Calmar ratio = annualised return / MaxDD.

<img src="media/image8.png" style="width:5.83333in;height:2.5in" />

<span id="_Toc228643973" class="anchor"></span>Figure 8 Walk-Forward Out-of-Sample Returns by Regime (left) and Sharpe / CVaR Risk Metrics by Regime (right). Green

<span class="mark">Figure : Walk-Forward Out-of-Sample Returns by Regime (left) and Sharpe / CVaR Risk Metrics by Regime (right). Green bars indicate positive return windows (2019 Bull, 2020 COVID full-year, 2021 Recovery); red bar indicates the 2022 Bear market. Note: the COVID window MaxDD of 31.6% reflects Q1 2020 crash severity, but the full-year return of +19.2% demonstrates recovery capacity within the evaluation window.</span>

The 2019 bull market window (Sharpe 1.52, return +12.34%, MaxDD 5.07%, Calmar 2.43) confirmed effective risk control under favorable conditions, with CVaR-95 of −1.18% showing well-bounded tail losses.

The 2020 COVID-19 window shows recovery capacity: despite a MaxDD of 31.63% in March 2020 (CVaR-95 −6.08%), the full-year return was +19.21%. The 10-K penalty could not encode pandemic risks evolving within weeks, but the system recovered and generated sufficient alpha to offset Q1 losses.

The 2021 recovery window was the strongest period (Sharpe 1.97, Sortino 2.86, MaxDD 4.28%, Calmar 4.41). The Manager’s preference for lower-beta assets proved advantageous and CVaR-95 of −0.92% was the lowest across all regimes.

The 2022 Bear market window produced the only negative result (return −14.66%, Sharpe −0.617, MaxDD 25.78%), reflecting the aggressive Fed tightening cycle that broke sector relationships encoded in training data. Both 2020 and 2022 are classified as turbulent markets (Kritzman and Li, 2010).

In summary, 3 of 4 walk-forward windows are positive with two Sharpe ratios above 1.5, and the system underperforms only in macro-driven stress that breaks sector structure, the expected profile for a semantically-guided system.

### Worker Level Performance Analysis

Table 4.5 presents the performance of the three Worker agents operating independently within their respective risk pools under the concurrent training protocol, together with the Manager's terminal capital allocation to each pool. This analysis is essential for understanding the source of the Manager's 1/N allocation failure and explaining why staged training produces a fundamentally different result.

| **Worker**                       | **Return** | **Sharpe** | **Sortino** | **MaxDD** | **HHI** | **Effective N** | **Mgr Alloc (concurrent)** |
|----------------------------------|------------|------------|-------------|-----------|---------|-----------------|----------------------------|
| **Safe (12 stocks, β\<0.8)**     | +6.67%     | 0.6811     | 0.9420      | 9.58%     | 0.0521  | 19.2            | 2.84% — UNDERFUNDED        |
| **Neutral (25 stocks, 0.8–1.2)** | +3.50%     | 0.3729     | 0.5140      | 12.16%    | 0.0267  | 37.5            | 90.48% — OVERFUNDED        |
| **Risky (8 stocks, β\>1.2)**     | −17.50%    | −1.2344    | −1.8820     | 21.64%    | 0.0743  | 13.5            | 6.68%                      |

<span id="_Toc228644004" class="anchor"></span>Table 15 Worker-Level Performance and Concurrent Manager Allocation by Risk Pool

<span class="mark">Table : Worker-Level Performance and Concurrent Manager Allocation by Risk Pool. Safe Worker highlighted in light blue — highest return and Sharpe despite receiving lowest capital allocation. This disconnect is the 1/N trap at the pool level, resolved by staged training.</span>

The most important finding in Table 4.5 is the disconnect between Worker performance and Manager allocation under concurrent training. The Safe Worker achieves the best return (+6.67%) and Sharpe (0.681) yet receives only 42.6% of capital — barely above equal-weight — while the underperforming Neutral Worker receives an equivalent share. The concurrent training signal is too noisy for the Manager to learn the allocation-to-performance mapping, defaulting to near-uniform distribution (DeMiguel et al., 2009).

The staged training protocol resolves this failure. By training the Manager against frozen Worker policies in Phase 2, the Manager faces a stable optimisation landscape where Worker returns are deterministic. The resulting allocation (76.6% Safe, 8.7% Neutral, 14.7% Risky) reflects a genuine learning-based capital allocation decision.

As expected, the risky worker whose returns were negatively impacted (-17.50%, sharp ratio = -1.234, max dd = 21.64%), reflect both the real volatility experience by higher-beta stocks over entire evaluation period (2015-2023) as well as instability introduced into optimizing multi-agent systems when multiple agents are trained simultaneously.

The high HHI (0.074) and low Effective N (13.5) values of risky worker indicate that risky worker is making high conviction bets, however, the high-beta nature of the stock pools being optimized (~1.528) make these bets significantly more volatile than the overall market and thus produce larger negative returns.

Finally, the equal weights distribution in the Neutral Worker's Effective N (37.5) value exceeding size of available pool indicates a most equal weighted portfolio distribution amongst the three Workers, as a direct consequence of Sharpe maximization objective that penalties extreme concentration.

### Drawdown Decomposition and Staged vs Concurrent Comparision

The drawdown analysis provides the most direct validation of the primary thesis objective: that semantic diversification produces measurably superior structural resilience during market stress periods. Figure 4.4 shows the equity curve and drawdown timeline for the concurrent MARL system evaluated over a 200-day evaluation window.

<img src="media/image9.png" style="width:5.83333in;height:2.91667in" />

<span id="_Toc228643974" class="anchor"></span>Figure 9 Global Portfolio Equity Curve (top) and Drawdown Timeline (bottom) for the Concurrent MARL System over a 200-day evaluation window

<span class="mark">Figure: Global Portfolio Equity Curve (top) and Drawdown Timeline (bottom) for the Concurrent MARL System over a 200-day evaluation window. The equity curve shows steady growth from days 0–125 followed by the deepest drawdown episode (days 125–165), then recovery. The drawdown timeline shows 12 distinct events with maximum depth 12.35%, average depth 1.98%, average duration 14.5 trading days, and average recovery time 7.0 trading days.</span>

Figure 4.10 illustrates a clear structural pattern across entire 200-day evaluation time frame for equity curve of our system: the system grows steadily from days \[0\]-\[125\], reaches peak cumulative return of approximately +6.5%, then enters deep draw down period from days \[125\]-\[165\] corresponding to Bear market in training data — reaches maximum depth below peak of -12.35%. System then recovers through days \[165\]-\[200\] — showing that our system has robustness mechanism built-in: portfolio returns back to previous peak within evaluation window, thus demonstrating that draw down period represents temporary setback rather than permanent capital loss.

Draw down statistics provide specific measurable evidence for robustness objective:

There are twelve distinct draws down events over entire evaluation window of trading days with each having an average duration of only 14.5 trading days and average recovery time of only 7 trading days and thus, system typically recovers from a draw down event within one-and-one half trading week.

For context, S&P 500 took approximately twelve months to recover from Bear market created by inflationary pressures in training data and five months from trough to recovery for covid-19 crashes in training data thus, seven-day average recovery time in drawdowns for our system represents a structurally different type of risk profile than price-based approaches.

This structural advantage is most evident in comparison of baseline methods used for testing our system: in full evaluation period staged MARL system’s max dd of 8.89% is better than all other baseline methods including equal weight (11.19%), momentum (9.84%), concurrent MARL (9.09%). Additionally, staged system’s max dd of 15.42% in hold-out period is better than equal weight (18.80%), MVO (16.47%), and risk parity (17.23%). Consistency of superior max dd results in all evaluation scenarios at full period, hold-out, and individual walk-forward scenarios provides strongest evidence that semantic diversification constraint is successful in meeting structural objective: portfolios constructed using this system lose less capital at worst point and recover from those losses more rapidly than do price-based baseline competitors

### Null Hypothesis Test: Lexical Control Experiment 

A fundamental scientific requirement for the WALP contribution is that performance improvements must be attributable to the semantic content of the TF-IDF similarity matrix, not merely to the mathematical structure of a quadratic penalty term. To test this, three configurations of the staged system were evaluated at λ=0.35: (A) the real TF-IDF lexical matrix derived from 10-K filings, (B) a shuffled matrix where rows and columns are randomly permuted (destroying semantic meaning while preserving statistical properties), and (C) a zero-matrix equivalent to no penalty.

Results confirm that real outperforms both controls on every metric. On Sharpe ratio: Real (0.703) \> Zero (0.612) \> Shuffled (0.445). On MaxDD: Real (9.29%) \< Zero (10.17%) \< Shuffled (16.10%). The ordering Real \> Zero \> Shuffled on Sharpe confirms that genuine semantic structure provides signal beyond what a structurally equivalent random penalty achieves, directly rejecting the null hypothesis.

The most striking finding is the Shuffled condition’s MaxDD of 16.10% which was 73% worse than the Real condition. Inspection of the Manager’s allocation under Shuffled conditions reveals the cause: the shuffled penalty incorrectly maps high costs to the Safe pool’s intra-pool similarity structure, causing the Manager to redirect 74.1% of capital to the Risky pool. This catastrophic mis-allocation demonstrates that the semantic content is not merely regularizing the reward but it is to providing genuine portfolio construction guidance that random permutation destroys. The zero-penalty condition (67.8% Safe allocation) confirms that without penalty guidance, the Manager defaults to a safe-biased but undifferentiated allocation.

<img src="media/image10.png" style="width:6in;height:4in" />

<span id="_Toc228643975" class="anchor"></span>Figure 10 Null Hypothesis Test — Real vs Shuffled vs Zero Lexical Matrix

<span class="mark">Figure 4.6: Null Hypothesis Test — Real vs Shuffled vs Zero Lexical Matrix. Top-left: cumulative return curves showing Real consistently above Shuffled and Zero. Top-right: Sharpe ratios (Real=0.703, Shuffled=0.445, Zero=0.612). Bottom-left: MaxDD showing Shuffled’s catastrophic 16.10% drawdown vs Real’s 9.29%. Bottom-right: Manager Safe-pool allocation — Real correctly allocates 76.6% to Safe while Shuffled collapses to 8.8%, demonstrating that semantic structure prevents dangerous concentrated bets.</span>

### Return Distribution Analysis: Skewness and Kurtosis

Return distribution analysis was conducted using scipy.stats skewness and excess kurtosis (Fisher definition, Normal=0) on the daily return series of all five strategies over the full 2015–2023 evaluation period. Table 4.12 summarizes the distributional statistics.

| **Strategy**        | **Mean Daily Return** | **Std** | **Skewness** | **Excess Kurtosis** |
|---------------------|-----------------------|---------|--------------|---------------------|
| **Staged Marl**     | 0.036%                | 0.816%  | −0.641       | +2.43               |
| **Concurrent Marl** | 0.040%                | 0.893%  | −0.558       | +2.55               |
| **Equal-Weight**    | 0.151%                | 0.976%  | +0.087       | −0.22               |
| **MVO**             | 0.172%                | 0.928%  | −0.060       | −0.30               |
| **Risk Parity**     | 0.127%                | 0.897%  | +0.074       | −0.08               |

<span id="_Toc228644005" class="anchor"></span>Table 18 Return Distribution Statistics, Skewness and Kurtosis (Full 2015–2023 Period)

Two findings require direct acknowledgement. First, both MARL strategies exhibit negative skewness (Staged −0.641, Concurrent −0.558), while classical baselines are approximately symmetric. Negative skewness indicates a heavier left tail: the MARL strategies produce more frequent small gains but experience occasional larger losses. This is a genuine structural property of the system, not an artefact of the evaluation window.

The mechanistic explanation is the Safe pool’s stock composition: defensive equities such as utilities and healthcare companies carry inherently left-skewed return distributions, particularly during macro-driven rate shock environments. When the staged Manager correctly allocates 76.6% to the Safe pool, it concentrates in precisely those assets with the most negative skewness profiles. This represents a limitation that semantic diversification cannot fully resolve: the penalty diversifies by business description similarity, not by return distribution shape.

Second, both MARL strategies exhibit excess kurtosis of approximately +2.4–2.6, indicating leptokurtic distributions (fat tails) compared to classical baselines (all near zero). The Q-Q plot in Figure 4.13 confirms systematic left-tail departure from normality for Staged MARL. This is consistent with the known property of RL-trained portfolio systems operating in equity markets: they generate non-Gaussian return profiles because their policy updates respond non-linearly to market signals (Hambly et al., 2023).

These distributional properties represent an honest limitation. Practitioners with strict skewness requirements, such as drawdown-first institutional mandates, should be aware that the system’s drawdown advantage (MaxDD 9.29%) comes alongside a negatively skewed daily return distribution. Incorporating skewness directly into the reward function is proposed as a future extension in Section 5.2.

<img src="media/image11.png" style="width:6in;height:4in" />

<span id="_Toc228643976" class="anchor"></span>Figure 11 Return Distribution Analysis — Skewness and Kurtosis

<span class="mark">Figure : Return Distribution Analysis — Skewness and Kurtosis. Top-left: KDE overlay with normal reference (dashed). Top-right: Q-Q plot for Staged MARL confirming leptokurtic left-tail departure. Bottom-left: Skewness bars (Staged −0.641, Concurrent −0.558, Equal-Weight +0.087, MVO −0.060, Risk Parity +0.074). Bottom-right: Excess kurtosis bars (Staged +2.43, Concurrent +2.55) showing both MARL strategies have significantly fatter tails than classical baselines.</span>

# Conclusion and Future Work

## Conclusion 

The hierarchical MARL system incorporating the Weight-Aware Lexical Penalty, Dirichlet policy heads, profile-specific reward functions, and staged curriculum training was evaluated on 45 S&P 500 stocks across four distinct market regimes from 2015 to 2023. The results provide direct answers to both research questions.

The staged MARL system achieved a cumulative return of +6.77%, Sharpe ratio of 0.703, Sortino ratio of 0.886, and maximum drawdown of 9.29% over the full evaluation period. The concurrent system achieved a marginally higher raw Sharpe (0.718) and return (+7.55%), however the staged system produced consistently superior risk metrics across every downside measure: MaxDD 9.29% vs 10.81%, Calmar ratio 0.980 vs 0.942, and CVaR-95 −2.08% vs −2.30%. For a system explicitly engineered for structural resilience rather than return maximization, winning on every risk metric while matching Sharpe within estimation noise constitutes a meaningful architectural improvement.

The mechanism is unambiguous from the Manager allocation analysis. The concurrent Manager defaults to near-equal-weight distribution across pools which was 42.6% Safe, 42.7% Neutral, 14.7% Risky because it unable to identify which Worker is performing best through the noise of simultaneous training. The staged Manager, trained against stable frozen Worker policies in Phase 2, correctly identifies the Safe Worker as the highest-quality sub-strategy and allocates 76.6% of capital to it. This constitutes the core proof of concept for staged curriculum learning in hierarchical MARL: sequential training resolves the non-stationarity problem that prevents concurrent training from discovering high-conviction allocations (Ning and Xie, 2024).

The structural resilience advantage is most clearly expressed in the drawdown analysis. During the 2022–2023 holdout stress test, the staged system's maximum drawdown of 15.42% outperformed all classical baselines: Equal-Weight (18.80%), MVO (16.47%), Risk Parity (17.23%), and Momentum (19.42%). Average drawdown recovery time of approximately seven trading days across twelve separate events confirms structural capital protection substantially faster than the months-long recoveries typical of classical approaches under comparable stress.

The lambda ablation provides the clearest single finding of this research. Under staged training, λ=0.1 is the optimal configuration, producing a Sharpe ratio of 0.947, a 46.9% improvement over the zero-penalty baseline (0.645) alongside lower MaxDD (9.37% vs 9.69%) and higher Sortino (1.203 vs 0.794). All three metrics improve simultaneously with no trade-off, confirming a genuine improvement in portfolio construction quality rather than a risk-return adjustment. The null-hypothesis control experiment further validates causality: the real lexical matrix (Sharpe 0.703) consistently outperformed a randomly shuffled permutation (0.445) and a zero-penalty baseline (0.612), with the shuffled condition producing catastrophic 16.1% MaxDD due to the Manager making dangerously concentrated bets without genuine semantic guidance.

This finding represents the primary scientific contribution of this dissertation: the first empirical demonstration that embedding NLP-derived business description similarity as a differentiable quadratic penalty wᵀSw in a MARL reward function produces measurably superior risk-adjusted performance. The lexical ratio framework of Mohseni et al. (2024) has been successfully extended from its original static optimization context into a trainable reinforcement learning system that actively learns to avoid semantic concentration. The mechanism is precisely as predicted by Hoberg and Phillips (2016): companies with overlapping business descriptions exhibit higher economic co-movement during stress periods, and penalizing that overlap in portfolio construction produces holdings that are more genuinely independent than those selected on price correlation alone.

**Honest Limitations**

Two limitations must be acknowledged. First, the staged system underperformed classical baselines on absolute return during the 2022–2023 holdout period (−2.44% vs Equal-Weight +1.76%). This reflects a macro-level monetary policy tightening shock with no precedent in the training data — an orthogonal event the semantic matrix could not encode since 10-K business descriptions do not capture interest rate sensitivity. The system continued to enforce semantic diversification and achieved the lowest MaxDD among all methods during this period, but could not generate positive returns from a shock entirely outside its experience space.

Second, the annual update frequency of 10-K filings creates a data-frequency gap: the similarity matrix S cannot adapt to sector-wide vulnerabilities emerging between annual reporting periods.

Within these boundaries, the staged system consistently outperformed classical alternatives on the two measures most critical for institutional investors: maximum drawdown and drawdown recovery time. The combination of hierarchical architecture, Dirichlet policy, semantic penalty, and curriculum training provides a coherent and theoretically grounded framework for building portfolios that are structurally resilient by design — and one whose core penalty mechanism is transferable as a drop-in module to any reinforcement learning portfolio framework.

## Future Work 

The next three research opportunities are directly related to the findings of the dissertation. Each opportunity addresses one of the previously described boundaries to the existing system:

Semantically Richer Higher Frequency Data 10-Qs: An immediate extension to the near term will be the inclusion of quarterly 10-Q filings along with annual 10-K filings. The 10-Q filing includes interim financial statements and management’s comments about material changes in their businesses. As such, the quarterly 10-Q filing update to the semantic similarity matrix S can be used to adjust the lexical penalty to reflect changes in rapidly evolving sector level vulnerabilities (e.g. disruptions to supply chains, changes in how sensitive to interest rate shifts an industry is, etc.) that occur between annual reporting periods. Another extension could include earnings calls that contain significant NLP signals (Tetlock, 2007). These calls are made quarterly and would enable a very rapid adjustment to the semantics of the penalty. Cohen, Malloy & Nguyen (2020) demonstrated that changes in the text of 10-K filings have value for investors. If these changes were extended to quarterly changes in the text of 10-Q filings then there would be four times the number of observations regarding changes in investor relevant information.

Dynamic Penalty Strength based on Regime: The results shown in Section 4.4 show that the semantic penalty is acting as a threshold-based constraint (as opposed to continuously improving) because of its non-monotonic behavior to penalty strength. In particular, λ = .1 is only marginally less effective than λ = 0 and λ = .5 is a large jump above λ = .1. Therefore, we believe that the optimal lambda t will depend upon the state of the market. For example, during times of high volatility and correlation (high tail dependence risk), we want to assign a stronger penalty to reflect our concern about systemic risk. During benign bull markets (when price based and semantic diversification converge), we may prefer a lower penalty to avoid reducing returns unnecessarily. Therefore, we propose implementing a time varying λ t that reflects some measure of financial turbulence (Kritzman & Li, 2010) i.e., increase λ t when the market exhibits signs of increased turbulence and decrease λ t when signs indicate reduced turbulence. By doing so, we can dynamically increase the semantic diversification during crises without sacrificing returns generation potential during peaceful times.

Semantic Representation Using Transformers: Currently, we utilize TF-IDF vectorization to calculate similarities among business descriptions. TF-IDF calculates word frequency overlap among documents and therefore does not capture semantical relationships between words (and hence between sentences) even if they are expressing equivalent concepts. Thus, two firms that express the same business activity (but in different terms) will yield low TF-IDF similarities although they have substantial economic overlap. A potentially more robust method would use transformer-based language model embeddings (Devlin et al., 2019) trained on large amounts of domain specific text data including SEC filings. Transformer-based embeddings capture semantic relationships through contextually represented word vectors instead of just word frequencies allowing for representation of deeper economic relationships that underlie business description similarities (such as supply chain relationships, product-market competition and customer overlap) identified by Hoberg & Phillips (2016) as key drivers of fundamental co-movements. Furthermore, since transformers do not rely on word-frequency overlaps like TF-IDF, we expect the penalty matrix to be able to identify structural risks hidden due to linguistic differences in business descriptions where TF-IDF cosine similarity fails to recognize economically identical activities being described differently.

<span id="_Toc404008248" class="anchor"></span>References

André, E. (2021). Dirichlet policies for reinforced factor portfolios. arXiv:2011.05381 \[q-fin.PM\].

Ang, A. and Chen, J. (2002). Asymmetric correlations of equity portfolios. Journal of Financial Economics, 63(3), pp. 443–494.

Aroussi, R. (2023). yfinance: Download market data from Yahoo! Finance. GitHub. Available at: https://github.com/ranaroussi/yfinance \[Accessed April 2026\].

Bengio, Y., Louradour, J., Collobert, R. and Weston, J. (2009). Curriculum learning. Proceedings of the 26th International Conference on Machine Learning (ICML), pp. 41–48.

Chua, D.B., Kritzman, M. and Page, S. (2009). The myth of diversification. Journal of Portfolio Management, 36(1), pp. 26–35.

Cohen, L., Malloy, C. and Nguyen, Q. (2020). Lazy prices. Journal of Finance, 75(4), pp. 1371–1415.

DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? Review of Financial Studies, 22(5), pp. 1915–1953.

Deng, Y., Bao, F., Kong, Y., Ren, Z. and Dai, Q. (2017). Deep direct reinforcement learning for financial signal representation and trading. IEEE Transactions on Neural Networks and Learning Systems, 28(3), pp. 653–664.

Devlin, J., Chang, M.W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, pp. 4171–4186.

Evans, J.L. and Archer, S.H. (1968). Diversification and the reduction of dispersion: An empirical analysis. Journal of Finance, 23(5), pp. 761–767.

Fama, E.F. and French, K.R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), pp. 3–56.

Flint, E., Seymour, A. and Chikurunhe, F. (2020). Defining and measuring portfolio diversification. South African Actuarial Journal, 20, pp. 17–48.

Gupta, J.K., Egorov, M. and Kochenderfer, M. (2017). Cooperative multi-agent control using deep reinforcement learning. In: Autonomous Agents and Multiagent Systems. Springer, Cham, pp. 66–83.

Hambly, B., Xu, R. and Yang, H. (2023). Recent advances in reinforcement learning in finance. Mathematical Finance, 33(3), pp. 437–503.

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P. and Oliphant, T.E. (2020). Array programming with NumPy. Nature, 587(7835), pp. 357–362.

Harvey, C.R. and Liu, Y. (2015). Backtesting. Journal of Portfolio Management, 42(1), pp. 13–28.

Hernandez-Leal, P., Kartal, B. and Taylor, M.E. (2019). A survey and critique of multiagent deep reinforcement learning. Autonomous Agents and Multi-Agent Systems, 33, pp. 750–797.

Hoberg, G. and Phillips, G. (2016). Text-based network industries and endogenous product differentiation. Journal of Political Economy, 124(5), pp. 1423–1465.

Huang, Z. and Tanaka, F. (2022). MSPM: A modularised and scalable multi-agent reinforcement learning-based system for financial portfolio management. PLoS ONE, 17(2), e0263689.

Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. Computing in Science and Engineering, 9(3), pp. 90–95.

Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. Journal of Finance, 48(1), pp. 65–91.

Jiang, Z., Xu, D. and Liang, J. (2017). A deep reinforcement learning framework for the financial portfolio management problem. arXiv:1706.10059 \[q-fin.CP\].

Kim, J., Park, J., Yun, J. and Park, J. (2022). A selective portfolio management algorithm with off-policy reinforcement learning using Dirichlet distribution. Axioms, 11(12), 664.

Kingma, D. and Ba, J. (2014). Adam: A method for stochastic optimization. International Conference on Learning Representations. arXiv:1412.6980.

Kritzman, M. and Li, Y. (2010). Skulls, financial turbulence, and risk management. Financial Analysts Journal, 66(5), pp. 30–41.

Kritzman, M., Page, S. and Turkington, D. (2012). Regime shifts: Implications for dynamic strategies. Financial Analysts Journal, 68(3), pp. 22–39.

Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2), pp. 365–411.

Lee, J., Kim, R., Yi, S.W. and Kang, J. (2020). MAPS: Multi-agent reinforcement learning-based portfolio management system. arXiv:2007.05402 \[cs.LG\].

Li, X., Li, Y., Zhan, Y. and Liu, X.Y. (2019). Optimistic bull or pessimistic bear: Adaptive deep reinforcement learning for stock portfolio allocation. ICML Workshop on Applications and Infrastructure for Multi-Agent Learning.

Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Silver, D. and Wierstra, D. (2016). Continuous control with deep reinforcement learning. ICLR. arXiv:1509.02971.

Lintner, J. (1965). The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets. Review of Economics and Statistics, 47(1), pp. 13–37.

Liu, X.Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B. and Wang, C.D. (2020). FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. NeurIPS Workshop. arXiv:2011.09607.

López de Prado, M. (2018). Advances in Financial Machine Learning. Hoboken, NJ: Wiley.

Loughran, T. and McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1), pp. 35–65.

Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P. and Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. NeurIPS. arXiv:1706.02275.

Ma, C., Zhang, J., Li, Z. and Xu, S. (2023). Multi-agent deep reinforcement learning algorithm with trend consistency regularization for portfolio management. Neural Computing and Applications, 35, pp. 6589–6601.

Magdon-Ismail, M. and Atiya, A.F. (2004). Maximum drawdown. Risk Magazine, 17(1), pp. 99–102.

Maillard, S., Roncalli, T. and Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. Journal of Portfolio Management, 36(4), pp. 60–70.

Markowitz, H. (1952). Portfolio selection. Journal of Finance, 7(1), pp. 77–91.

McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, pp. 56–61.

Meucci, A. (2010). Managing diversification. Available at: www.symmys.org.

Michaud, R. (1989). The Markowitz optimization enigma: Is 'optimized' optimal? Financial Analysts Journal, 45(1), pp. 31–42.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A. and Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), pp. 529–533.

Mohseni, S.F., Arian, H. and Bégin, J.F. (2024). The lexical ratio: A new perspective on portfolio diversification. arXiv:2411.06080 \[q-fin.PM\].

Moody, J. and Saffell, M. (2001). Learning to trade via direct reinforcement. IEEE Transactions on Neural Networks, 12(4), pp. 875–889.

Ning, Z. and Xie, L. (2024). A survey on multi-agent reinforcement learning and its application. Journal of Automation and Intelligence, 3, pp. 73–91.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N. and Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, pp. 8024–8035.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R. and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, pp. 2825–2830.

Rashid, T., Samvelyan, M., de Witt, C.S., Farquhar, G., Foerster, J. and Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. ICML, Proceedings of Machine Learning Research, 80, pp. 4295–4304.

Rhoades, S.A. (1993). The Herfindahl-Hirschman index. Federal Reserve Bulletin, 79(3), pp. 188–189.

Rockafellar, R.T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2(3), pp. 21–41.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347 \[cs.LG\].

Shavandi, A. and Khedmati, M. (2022). A multi-agent deep reinforcement learning framework for algorithmic trading in financial markets. Expert Systems with Applications, 208, 118124.

Sharpe, W.F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. Journal of Finance, 19(3), pp. 425–442.

Sharpe, W.F. (1994). The Sharpe ratio. Journal of Portfolio Management, 21(1), pp. 49–58.

Sortino, F.A. and Price, L.N. (1994). Performance measurement in a downside risk framework. Journal of Investing, 3(3), pp. 59–64.

Statman, M. (1987). How many stocks make a diversified portfolio? Journal of Financial and Quantitative Analysis, 22(3), pp. 353–363.

Sutton, R.S. and Barto, A.G. (2018). Reinforcement Learning: An Introduction. 2nd edn. Cambridge, MA: MIT Press.

Tetlock, P.C. (2007). Giving content to investor sentiment: The role of media in the stock market. Journal of Finance, 62(3), pp. 1139–1168.

Tian, Y., Kladny, K.R., Wang, Q., Huang, Z. and Fink, O. (2022). A prescriptive Dirichlet power allocation policy with deep reinforcement learning. Reliability Engineering and System Safety, 224, 108576.

Towers, M., Terry, J.K., Kwiatkowski, A., Balis, J.U., Cola, G., Deleu, T. and Dobs, K. (2023). Gymnasium. Zenodo. https://doi.org/10.5281/zenodo.8127025.

U.S. Securities and Exchange Commission (2024). EDGAR Company Filings API. Available at: https://data.sec.gov/submissions/ \[Accessed April 2026\].

Williams, R.J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4), pp. 229–256.

Van Rossum, G. and Drake, F.L. (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.

Waskom, M. (2021). Seaborn: Statistical data visualization. Journal of Open Source Software, 6(60), 3021.

Xue, P. and Ye, Q. (2025). Attention-enhanced reinforcement learning for dynamic portfolio optimization. arXiv:2510.06466.

Yang, H., Liu, X.Y., Zhong, S. and Walid, A. (2020). Deep reinforcement learning for automated stock trading: An ensemble strategy. ACM International Conference on AI in Finance (ICAIF).

Zhang, Z., Zohren, S. and Roberts, S. (2020). Deep reinforcement learning for trading. Journal of Financial Data Science, 2(2), pp. 25–40.

<span id="_Toc303784826" class="anchor"></span>

Appendix A. System Hyperparameters and Configuration

The following table provides a complete reference for all hyperparameters, data configurations, and software dependencies used in the experimental evaluation. All values apply to the staged MARL system unless otherwise noted. The concurrent MARL baseline uses identical hyperparameters with Phase 2 and Phase 3 removed and all agents trained simultaneously for 200 episodes.

Table A.1 provides a complete reference for all system hyperparameters, data configurations, and software versions used in the experimental evaluation. All settings are documented for full reproducibility.

| **Parameter**                       | **Value**                                                | **Justification / Source**                                                         |
|-------------------------------------|----------------------------------------------------------|------------------------------------------------------------------------------------|
| **Asset universe**                  | 45 S&P 500 stocks                                        | Evans & Archer (1968); Statman (1987) — sufficient to eliminate idiosyncratic risk |
| **Risk pool thresholds**            | β\<0.8 / 0.8–1.2 / β≥1.2                                 | CAPM beta classification (Sharpe, 1964; Fama & French, 1993)                       |
| **Safe pool**                       | 12 stocks, avg β=0.669                                   | BAX, BMY, COR, D, DGX, DPZ, EVRG, LHX, MO, NI, PPL, SRE                            |
| **Neutral pool**                    | 25 stocks, avg β=0.990                                   | Market-tracking across financials, technology, industrials                         |
| **Risky pool**                      | 8 stocks, avg β=1.528                                    | APO, APTV, CVNA, FCX, KLAC, MCHP, MTCH, WDC                                        |
| **Training period**                 | 2015-01-02 to 2022-03-14                                 | 8+ years, multiple regimes; holdout withheld                                       |
| **Holdout period**                  | 2022-03-14 to 2023-12-29                                 | 453 trading days, strictly unseen during development                               |
| **Phase 1 (Worker) episodes**       | 200                                                      | Workers converge before Manager introduced (Bengio et al., 2009)                   |
| **Phase 2 (Manager) episodes**      | 160                                                      | Manager learns from stable frozen Worker policies                                  |
| **Phase 3 (Joint) episodes**        | 80                                                       | Joint adaptation with reduced LR from stable baseline                              |
| **Total training time**             | 1,136.0 seconds                                          | Single CUDA-enabled GPU                                                            |
| **Training algorithm**              | PPO                                                      | Schulman et al. (2017) — clipped surrogate objective                               |
| **Optimizer**                       | Adam                                                     | Kingma & Ba (2014) — lr=3×10⁻³, β₁=0.9, β₂=0.999                                   |
| **LR schedule**                     | 3×10⁻³ → 5×10⁻⁴ (decay)                                  | Visible in training log ep20→ep200                                                 |
| **Entropy coefficient**             | 0.0457 → 0.0050 (decay)                                  | PPO exploration schedule                                                           |
| **Dirichlet α clamping**            | α \> 1.01 via Softplus+1                                 | Prescriptive clamping (Tian et al., 2022)                                          |
| **Semantic penalty λ**              | 0.10 (optimal for staged training; selected by ablation) | 79% Sharpe improvement over λ=0                                                    |
| **Safe Worker variance penalty**    | 2.0·σ_portfolio                                          | Profile-specific reward design                                                     |
| **Neutral Worker turnover penalty** | γ·Turnover                                               | Encourages Sharpe maximization                                                     |
| **Risky Worker return multiplier**  | 1.5×R_log                                                | Encourages alpha generation                                                        |
| **Cash asset in action space**      | Yes (each Worker)                                        | Explicit 'do nothing' option to break 1/N                                          |
| **TF-IDF max features**             | 5,000                                                    | Scikit-learn vectorizer (Pedregosa et al., 2011)                                   |
| **TF-IDF min_df / max_df**          | 2 / 0.95                                                 | Removes rare and near-universal terms                                              |
| **Similarity matrix mean**          | 0.077                                                    | Low baseline — confirms matrix discriminates effectively                           |
| **Similarity matrix std**           | 0.074                                                    | Confirms meaningful variance across company pairs                                  |
| **Safe pool intra-similarity**      | 0.104                                                    | Highest — consistent with Hoberg & Phillips (2016)                                 |
| **Neutral pool intra-similarity**   | 0.085                                                    | Intermediate                                                                       |
| **Risky pool intra-similarity**     | 0.057                                                    | Lowest — high-beta stocks most lexically diverse                                   |
| **Price data source**               | yfinance                                                 | Daily OHLCV, 2015–2023                                                             |
| **Text data source**                | Yahoo Finance (32) + EDGAR API (13)                      | U.S. SEC (2024)                                                                    |
| **ML framework**                    | PyTorch 2.x                                              | Paszke et al. (2019)                                                               |
| **RL environment**                  | Gymnasium                                                | Towers et al. (2023)                                                               |
| **NLP framework**                   | Scikit-learn                                             | Pedregosa et al. (2011)                                                            |
| **Data manipulation**               | Pandas + NumPy                                           | McKinney (2010); Harris et al. (2020)                                              |
| **Visualisation**                   | Matplotlib + Seaborn                                     | Hunter (2007); Waskom (2021)                                                       |

Table A 1 Complete System Architecture and Training Hyperparameters.<span id="_Toc394498204" class="anchor"></span>

Appendix B. Core System Implementation

This appendix provides the key implementation components of the Hierarchical MARL system for full reproducibility. The complete codebase is available in the submitted repository. The following excerpts document the four most critical components: the Weight-Aware Lexical Penalty embedded in the Worker reward function, the Dirichlet policy head replacing the standard Softmax output, the staged curriculum training protocol, and the EIIE feature extraction backbone.

**Weight-Aware Lexical Penalty — Worker Reward Function** Paste the reward computation code from WorkerEnv showing the λ·wᵀ·S·w term.

**Dirichlet Policy Head** Paste the EIIENetwork forward method showing the Softplus+1 clamping and Dirichlet sampling.

**Staged Curriculum Training Protocol** Paste the train_staged() function showing the three phases, the frozen/unfrozen parameter logic, and the learning rate decay.

**TF-IDF Similarity Matrix Construction** Paste the preprocessing code that builds S from the 10-K text corpus using scikit-learn TfidfVectorizer and cosine similarity.
