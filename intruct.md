# Research Proposal: Robust Multivariate Time Series Forecasting via Density-Filtered Topological Data Analysis and Advanced Optimization

## 1. Research Topic Explanation
### 1.1 Background & Motivation
In previous work, Topological Data Analysis (TDA) was successfully applied to 1-dimensional time series forecasting. By embedding a 1D series into a multidimensional point cloud using Taken's embedding theorem, persistent homology captured the underlying geometric shape of the dynamical system, improving the accuracy of LSTM and XGBoost models. 

However, real-world systems are rarely 1-dimensional. Transitioning to **Multivariate Time Series (MTS)** introduces severe complexities:
1.  **Cross-Variable Interactions:** Variables interact across both time and space. Standard independent embeddings fail to capture the synchronization or decoupling of different features.
2.  **Vulnerability to Outliers:** In higher dimensions, the probability of sampling outliers increases. TDA is notoriously sensitive to outliers; a single anomalous data point can artificially create or destroy topological features (loops, voids), leading to brittle persistence diagrams and degraded model performance.
3.  **Parameter Sensitivity:** The topological structure is heavily dictated by the embedding parameters (time delay $\tau$, embedding dimension $d$, sliding window $L$). Relying on static heuristics for these parameters limits predictive capacity.

### 1.2 Proposed Solution
This research proposes a robust TDA-MTS forecasting framework:
* **Joint Phase-Space Reconstruction:** Creating a unified point cloud that captures the dependencies between multiple variables using cross-correlation or Mahalanobis distance, rather than simple Euclidean metrics.
* **Density-Based Outlier Filtration:** Utilizing nonparametric density estimation (e.g., Distance-to-Measure or Kernel Density Estimation) to quantify the density around each point. Low-density points (outliers) are penalized or filtered out before computing the Vietoris-Rips complex, ensuring the resulting topology represents the true system dynamics.
* **Advanced Optimization Techniques (AOT):** Utilizing Bayesian Optimization or Particle Swarm Optimization (PSO) to dynamically search for the optimal point-cloud parameters ($\tau, d, L$), using the validation loss of the downstream forecasting model as the objective function.

---

## 2. Recommended Dataset: Quantitative Finance
**Dataset Name:** S&P 500 Multivariate Financial Time Series (OHLCV + Macro Indicators)
**Source:** Yahoo Finance API (`yfinance` in Python) or Kaggle.

### 2.1 Dataset Composition
To capture multidimensional interactions, the dataset should consist of daily data spanning 10-15 years, including:
1.  **Target Variable:** S&P 500 Daily Closing Price (or log returns).
2.  **Internal Market Features:** Open, High, Low, Volume (OHLCV).
3.  **Volatility Index:** VIX (captures market sentiment and fear).
4.  **Macroeconomic Indicators:** 10-Year Treasury Yield, US Dollar Index (DXY).
5.  **Sector Indices:** Tech (XLK), Financials (XLF), Energy (XLE) to capture cross-sector interactions.

### 2.2 Why this fits the objective:
* **Multidimensionality:** The target is influenced by a complex web of internal market mechanics and external macroeconomic forces.
* **Outlier Richness:** Financial markets contain natural outliers (market crashes, flash crashes, earnings surprises) making it the perfect testing ground for density-based outlier filtration.
* **Real-World Applicability:** Improving predictive accuracy in financial forecasting by even a fraction of a percent has immense real-world value.

---

## 3. Methodology: What to Do (Step-by-Step)

### Step 1: Data Preprocessing
* Fetch the multivariate financial dataset.
* Handle missing values and apply transformations (e.g., converting prices to log returns to achieve stationarity).
* Normalize the data using Min-Max or Standard scaling.

### Step 2: Multivariate Phase-Space Reconstruction
* Instead of embedding a 1D series, combine the $N$ variables into a joint trajectory matrix using a sliding window $L$ and time delay $\tau$.
* *Implementation:* Construct a point cloud where each point represents the state of all market variables at a specific time step.

### Step 3: Density Estimation & Outlier Filtration
* *Implementation:* Apply a Kernel Density Estimator (KDE) or compute the Distance-to-Measure (DTM) for every point in the reconstructed space.
* Establish a density threshold. Points falling below this threshold are flagged as homological noise/outliers.
* Apply a weighted filtration where the distance metric used in the Vietoris-Rips complex is scaled by the inverse of the point density.

### Step 4: Topological Feature Extraction
* Compute the persistent homology ($H_0, H_1, H_2$) using libraries like `ripser` or `GUDHI`.
* Vectorize the resulting persistence diagrams into Persistence Landscapes or Persistence Images using `persim`, so they can be fed into neural networks.

### Step 5: Predictive Modeling & AOT Loop
* Concatenate the vectorized TDA features with standard lag features.
* Feed the concatenated vector into an LSTM or N-BEATS architecture to predict the next time step (e.g., next day's S&P 500 return).
* **AOT Integration:** Wrap steps 2-5 in a Bayesian Optimization loop (using `Optuna` or `hyperopt`). Define the parameter search space for $\tau$ (delay), $d$ (dimension), and the KDE bandwidth. The AOT minimizes the LSTM validation RMSE.

---

## 4. List of Experiments

To rigorously validate the proposed methodology, the following experiments must be conducted and documented:

### Experiment 1: Baseline Performance Comparison
* **Objective:** Establish the superiority of the proposed model against standard methods.
* **Models to Compare:**
    1. Standard LSTM (No TDA features).
    2. Standard XGBoost (No TDA features).
    3. TDA-LSTM (Standard Euclidean TDA, no outlier filtration - replicating previous MTP1 approach on multivariate data).
    4. **Proposed:** Robust Density-Filtered TDA-LSTM.
* **Metrics:** RMSE, MAE, and Directional Accuracy (hit rate of predicting market direction).

### Experiment 2: Impact of Multidimensionality
* **Objective:** Prove that joint multidimensional embeddings capture more information than isolated 1D embeddings.
* **Setup:** * Run the forecasting model using TDA features extracted *only* from the target variable (1D S&P 500 Close).
    * Compare against TDA features extracted from the joint phase-space of all variables (OHLCV + VIX + Macro).

### Experiment 3: Robustness to Outliers (Ablation Study)
* **Objective:** Quantify the effectiveness of the Density-Based Outlier Filtration.
* **Setup:** * *Phase A:* Train and evaluate the model using standard TDA features (without density filtration).
    * *Phase B:* Train and evaluate using Density-Filtered TDA features.
    * *Phase C (Stress Test):* Artificially inject synthetic noise (extreme price spikes/crashes) into the training dataset. Measure the degradation in performance for both Phase A and Phase B models. The density-filtered model should show significantly less performance degradation.

### Experiment 4: Heuristic vs. AOT Parameter Optimization
* **Objective:** Demonstrate that Advanced Optimization Techniques yield better topological parameters than static rules of thumb (like False Nearest Neighbors or Mutual Information).
* **Setup:**
    * Select parameters ($\tau, d$) using standard heuristic methods.
    * Use Bayesian Optimization (via `Optuna`) to dynamically search for ($\tau, d, L$).
    * Compare the final validation loss and computational time between the two approaches. 

### Experiment 5: Feature Importance and Interpretability
* **Objective:** Understand the contribution of topological features.
* **Setup:** Use SHAP (SHapley Additive exPlanations) values on an XGBoost surrogate model to rank the importance of the standard lag features versus the 0-dimensional (clusters) and 1-dimensional (loops) topological features.