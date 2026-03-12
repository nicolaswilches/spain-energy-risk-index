**PROJECT: SPAIN ENERGY GRID RISK INDEX (MLOps Phase 1-3 Revised)**
You are my executing MLOps agent. Please read this entire architecture plan carefully before writing any code. We are building a Machine Learning pipeline to predict a continuous "Grid Risk Index" for the Spanish electrical grid (REE) using LightGBM. We have already executed a first iteration of stages 1 to 3, however, we need to execute modifications specified below.

## 1. Architectural Pivot (CRITICAL)

Based on API limitations and academic feedback, we are pivoting the project from an hourly forecast to a **Daily Forecast**. The model will predict a single Risk Index score (0 to 1) for tomorrow. To satisfy the requirement for independent variables, we will enrich the grid data with external weather and calendar features. Once revisions for stage 1 to 3 are complete, let's plan execution of the rest of the project, focusing on tracking experiements using MLFlow, building an API to host locally our trained model using FASTAPI, creating a Docker image, updating the image to GitHub using GitHub Actions and rendering the image using a platform such as Render.com. Finally as a nice add-in for the project, building a streamlit application that can request a prediction from our model and display it in a intuitive UI.

---

## 2. Project Structure Example (ie-mlops-nyc-taxis)

To have a clear picture of the structue of the final model, we have a guideline to follow. Refer to the folder (path below) to have a clear picture of the final deliverables of the project.
Sample final project structure: /Users/nicolaswilches/ds/projects/ie-mlops-nyc-taxis

---

## 3. Project Guidelines

Deliverable: (**GitHub Repository**)
Must include:

* Model training script (**train.py**, MLflow tracking)
* Serving API (**app.py**, FastAPI )
* Dockerfile (containerized service, Dockerfile)
* CI/CD workflow (.github /workflows/ ci-cd.yml )
  * Lint + test + build + deploy
* Deployment manifest (**render.yaml**)
  * Working online endpoint (Render.com or equivalent)
* **README.md** explaining setup, workflow, and usage
Ensure best practices and a proper project structure are used.

---

## 4. General Best Practices

1. **Modularization:** Break down code into smaller, reusable functions or classes.
2. **Configuration Management:** Use configuration files (config.yaml) or environment variables for settings.
3. **Logging and Monitoring:** Add logging (MLFlow) and possibly some basic monitoring.
4. **Error Handling:** Include comprehensive error handling and data validation.
5. **Dependency Management:** List all dependencies and versions for reproducibility.
6. **Documentation:** Include sufficient comments and documentation for maintainability.
7. **Code Quality:** Ensure the code is clean, readable, and follows PEP 8 standards.
8. **Testing:** Include unit tests to validate the functionality.
9. **Security:** Remove any sensitive or secret info (.env).

---

## 5. Revision of Phase 1 to 3

### 2. Phase 1: Data Engineering & API Integration

**Objective:** Fetch, clean, and merge three data streams into a single Daily Pandas/Polars DataFrame.
**Update:** We now possess a token provided by the REE to access their data. This might be useful for data fetching. The token is: "a90d42b9b4583a94529db4d25a7ad67c6c305833f42ea7716f45da6f6bf0bda7"

* **Stream A: REData API (Grid Actuals)**
  * *Endpoint:* `https://apidatos.ree.es/en/datos/...`
  * *Granularity:* `time_trunc=day`
  * *Required Data:* Actual Demand (MW), and Actual Generation by Technology (Wind, Solar, Hydro, Combined-Cycle, Nuclear).
  * *Required Data:* Day-Ahead Demand Forecast (MW).
* **Stream B: Open-Meteo API (Weather)**
  * *Endpoint:* `https://api.open-meteo.com/v1/forecast` (Historical)
  * *Location:* Coordinates for Madrid, Spain (approx. 40.41, -3.70).
  * *Required Data (Daily):* `temperature_2m_max`, `temperature_2m_min`, `windspeed_10m_max`, `shortwave_radiation_sum`, `precipitation_sum`.
* **Stream C: Calendar & Human Activity**
  * Use the Python `holidays` library (`holidays.Spain()`) to create an `Is_Holiday` boolean feature.
  * Create `Day_of_Week` (0-6), `Month` (1-12), and `Is_Weekend` (0/1).
  **Stream D: Additional features**
  * Consider any other features that might affect the energy demand/consumption in Spain and for which we could extract data under the same granularity and through an publicly avaialble API.

* **Output:** Save the merged, chronologically aligned dataset to `data/merged_daily_data.parquet` (do not commit this to Git).

### 3. Phase 2: Target Creation & Feature Engineering

**Objective:** Calculate the "Ground Truth" Risk Index ($Y$) and engineer the predictive features ($X$) without introducing data leakage.

* **Chronological Split (Strictly execute BEFORE PCA):**
  * *Train Set:* e.g., 2021-01-01 to 2023-12-31
  * *Validation Set:* e.g., 2024-01-01 to 2024-06-30
  * *Test Set:* e.g., 2024-07-01 to Present
* **Target Variable Creation ($Y$):**
    1. Calculate 3 core operational stress variables for all rows:
        * `Flexibility_Share` = (Combined-Cycle + Hydro) / Total Generation
        * `Demand_Forecast_Error` = Actual Demand - Forecasted Demand
        * `Net_Load` = Actual Demand - (Actual Wind + Actual Solar)
    2. Fit a `StandardScaler` and Principal Component Analysis (`PCA`) **ONLY** on the Train Set's 3 core factors.
    3. Extract the weights from PC1. Transform all sets (Train, Val, Test) using the fitted PCA to create a continuous `Risk_Index` (scaled 0 to 1).
    4. Determine thresholds for *Low*, *Medium*, and *High* risk based on percentiles in the Train set (e.g., 90th percentile = High).
* **Predictive Features ($X$):**
  * The Open-Meteo weather features, Calendar features, and REE Day-Ahead Forecast features for day $t$.
  * **Lagged Features:** `Risk_Index_Lag_1d` (Yesterday's Risk Index) and `Risk_Index_Lag_7d` (Risk Index exactly one week ago).

### 4. Phase 3: Model Training & Evaluation (Business-Aligned MLOps)

**Objective:** Train LightGBM using a pessimistic loss function and evaluate it using custom operational metrics. Track experiments with MLflow.

* **Model Training (Quantile Regression):**
  * A False Negative (missing a blackout) is 10x more expensive than a False Positive (wasting gas plant fuel). We cannot use standard MSE.
  * Define the LightGBM Regressor using **Quantile Regression** (`objective='quantile'`, `alpha=0.90`). This forces the model to be cautious and trace the upper boundaries of grid stress.
  * Tune hyperparameters (`reg_alpha`, `reg_lambda` for feature pruning) on the Validation Set.
* **Custom Evaluation Metrics (Test Set):**
  * **Metric A - Baseline-Relative RMSE:** Calculate standard RMSE for the LightGBM model and a Persistence Baseline model ($t_{+1} = t_{0}$). Report the percentage improvement (Skill Score).
  * **Metric B - Custom Cost Matrix:** For every *High-Risk* day the model misses (False Negative), add 10 penalty points. For every safe day the model falsely flags as *High-Risk* (False Positive), add 1 penalty point. Minimize this score compared to the baseline.
  * **Metric C - $F_3$ Score:** Calculate the $F_\beta$ score where $\beta=3$ for the *High* class, weighting Recall as 9 times more important than Precision.
* **Model predictions on test set**

* **Output:** Consolidate the current 3 notebooks in the 'notebooks' folder into a single notebook that contains a a detailed log of the work done in stages 1 to 3. The idea is to have an interactive step by step of the work contained in the .py scripts built to conduct a follow up and play with the model built. The notebook must contain these main sections and the correspondent subsections.
  * **Phase 1. Data Engineering & API Integration**
  * **Phase 2: Target Creation & Feature Engineering:** Displying the split in training, validation and test; feature engineering, PCA for 3 core operational stress variables and target variable creation; and predictive features.
  * **Phase 3: Model Training & Evaluation**
  * **Additional considerations for the notebook:**
    * Use plotly for visualizations with Arial font and consistent and meaningful color coding.
    * Add insightful visualizations for each stage of the notebook:
      * For phase 1 add add histograms to display distribution of our features.
      * For phase 2 add visualizations useful to interpet 3 core op. metrics and PCA.
      * For phase 3 add visualizations for model evaluation such as: Evaluation metrics evolution vs. number of model iterations, SHAP summary plot to prove to stakeholders that Weather (`temperature`, `windspeed`) and Calendar (`holidays`) features are actively driving the Risk Index predictions, feature importances.

---

## 6. Next Steps

Once the model has been trained and evaluated, plan de exection of the following stages:

* Dockerfile (containerized service, Dockerfile).
* CI/CD workflow (.github /workflows/ ci-cd.yml )
  * Lint + test + build + deploy
* Deployment manifest (**render.yaml**)
  * Working online endpoint (Render.com or equivalent)
* **README.md** explaining setup, workflow, and usage
Ensure best practices and a proper project structure are used.

Ask questions for the planning of these stages before executing them.

---

**Action Required:**
Confirm you understand this architecture. Once confirmed, begin executing the instructions provided above. After each Phase is completed, stop, test, verify work and notify on progress achieved before moving on.
