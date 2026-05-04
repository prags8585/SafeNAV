# 🛣️ Road Accident Severity Predictor
### Machine Learning Course — Semester Project Report

**Course**: Fundamentals of Machine Learning for Predictive Data Analytics  
**Textbook**: Kelleher, Mac Namee & D'Arcy — *The MIT Press (2015)*  
**Team**: [Your Name(s)]  
**Date**: May 2026

---

## 1. Problem Statement & Real-World Application

Every year, millions of road accidents occur across the United States. While many are minor fender-benders, others are fatal. The challenge faced by emergency responders, road safety agencies, and autonomous vehicle systems is the same:
> **Given the conditions at the time of an accident — the weather, time of day, road type, and visibility — how severe will the accident be?**

### Why Simple Statistics Fail
A common misconception is that single factors dictate severity (e.g., "accidents at night are always more fatal"). However, our raw data analysis reveals that human behavior compensates for obvious risks. For example, nighttime accidents are on average barely more severe than daytime accidents (19.7% vs 18.8% severe) because drivers slow down and drive more cautiously.

However, when multiple risk factors combine (e.g., **Nighttime + Highway Junction + Heavy Rain + No Traffic Signal**), the probability of a fatal crash spikes dramatically. Simple statistics fail to predict severity because they cannot map these complex, non-linear interactions. Machine Learning algorithms are required to automatically discover these deadly combinations.

### Real-World Applications
If deployed, this predictive model serves critical functions across multiple industries:
1. **Emergency Dispatch (Smart Triage)**: A 911 dispatcher system can query the model using real-time GPS and weather APIs. If the model outputs a high probability of a "Severity 4" (Fatal) accident based on the environment, the system can instantly pre-dispatch trauma units and helicopters before the severity is even confirmed by human callers, saving critical minutes in the "Golden Hour" of trauma medicine.
2. **Autonomous Vehicles (Tesla/Waymo)**: Self-driving software constantly assesses real-time environmental risk. If the model flags the current conditions as highly conducive to fatal accidents, the vehicle can automatically reduce its maximum speed and increase following distance.
3. **Dynamic Navigation (Google Maps/Waze)**: Routing algorithms can actively direct drivers away from rural junctions during dense fog if the model flags that specific route as high-risk under current weather conditions.

---

## 2. Dataset Overview

### Source
**US Accidents Dataset** — publicly available on Kaggle (~3.0 GB, 7.7 million records)

### Data Suitability
The dataset is incredibly robust for Machine Learning. Our sample analysis confirms:
*   **Scale**: Millions of records ensure models have enough data to find complex patterns without overfitting.
*   **Diversity**: 46 original features covering continuous variables (Temperature, Visibility), categorical (Weather Condition), and boolean (Traffic Signal, Junction).
*   **Target Variable (`Severity`)**: A 1-4 scale representing the impact of the accident. 

**The Class Imbalance Challenge**: The data is heavily imbalanced (~80% of accidents are Severity 2, while Severity 4 is < 3%). This accurately reflects the real world but poses a challenge for ML. We address this using **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure our models don't blindly predict "Severity 2" for every accident.

---

## 3. Phase-Wise Implementation Strategy

To satisfy the rubric requirements for comprehensive Data Understanding, Data Preparation, Modeling, and Evaluation, we have structured the project into six distinct phases.

### Phase 1: Pre-Cleaning Exploratory Data Analysis (EDA)
**Goal**: Analyze the raw dataset distributions before any cleaning or imputation.
*   **Graph 1**: A massive multi-plot canvas showing the raw distribution of all key features.
*   **Graph 2**: Target variable (`Severity`) distribution (highlighting the extreme class imbalance).
*   **Graph 3**: Accidents by Hour / Time of Day.
*   **Graph 4**: Severity vs. Weather Condition.
*   **Graph 5**: Geographic spread (Top 10 States).

### Phase 2: Data Preprocessing & Post-Cleaning EDA
**Goal**: Clean the data and visualize the effectiveness of the preprocessing.
*   **Cleaning Pipeline**: Drop high-null columns (e.g., `Wind_Chill` with 26% nulls), median impute missing weather values, scale numeric features using `StandardScaler`, and apply SMOTE to balance the target classes.
*   **Graph 1**: The new balanced `Severity` distribution post-SMOTE.
*   **Graph 2**: Pre vs. Post imputation distributions (e.g., Visibility) to prove imputation did not distort the underlying data.
*   **Graph 3**: Correlation heatmap of the finalized, scaled feature set.

### Phase 3: Decision Tree Model (Information-Based)
**Concept**: Learns by creating IF/THEN rules based on Information Gain. Highly interpretable and excellent at finding complex feature combinations (e.g., Night + Fog + Junction).
*   **Implementation**: Train `DecisionTreeClassifier`.
*   **Visualizations**: Plot the visual Decision Tree structure and a Feature Importance bar chart.
*   **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.

### Phase 4: Naïve Bayes Model (Probability-Based)
**Concept**: Uses Bayes' Theorem to calculate the posterior probability of a fatal accident. Very fast, but makes a "naïve" assumption that all environmental features (like rain and humidity) are statistically independent.
*   **Implementation**: Train `GaussianNB`.
*   **Visualizations**: Plot Class Prior Probabilities and Feature Likelihood Distributions.
*   **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.

### Phase 5: Logistic Regression Model (Error-Based)
**Concept**: Learns by assigning mathematical weights (coefficients) to every feature to minimize cross-entropy loss via gradient descent. Excellent at providing reliable, calibrated probability scores.
*   **Implementation**: Train multinomial `LogisticRegression`.
*   **Visualizations**: Plot Coefficient Weights (showing exactly what features push a score toward "Fatal") and the model's Learning Curve.
*   **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.

### Phase 6: Final Comparison & Conclusion
**Goal**: Compare the models and determine the safest choice for real-world deployment.
*   **Deliverables**: An Overlaid ROC Curve (all 3 models on one graph) and a side-by-side metric comparison table.

---

## 4. Evaluation Metrics Explained

Because our dataset is heavily imbalanced, using standard **Accuracy** is dangerously misleading. A model that blindly guesses "Non-Fatal" every time would be mathematically accurate but practically useless in emergency response. 

Therefore, our primary metrics for crowning a winning model are:
1.  **Recall (Sensitivity) on the Fatal Class**: Out of all the *actual* Fatal accidents, how many did the model catch? Missing a fatal accident is a catastrophic failure (False Negative). We prioritize maximizing this metric.
2.  **Precision**: Out of all the times the model predicted "Fatal", how many were actually fatal? We monitor this to ensure the model isn't triggering too many false alarms.
3.  **F1-Score**: The harmonic mean of Precision and Recall, providing a balanced view of the model's performance on the minority class.
4.  **ROC / AUC (Area Under the Curve)**: A visual representation of how well the model separates the Fatal class from the non-fatal classes across all threshold probabilities.

---

## 5. Conclusion & Next Steps
We will execute the 6-phase plan in individual Jupyter Notebooks. By testing an Information-based model, a Probability-based model, and an Error-based model against the exact same test dataset, we will definitively prove which paradigm is best suited for predicting the complex reality of traffic accidents.
