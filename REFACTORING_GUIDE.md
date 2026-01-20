# Election Prediction Pipeline Refactoring Guide

## Overview

This document summarizes the refactoring from a Texas-specific gubernatorial model to a generalizable multi-race framework.

**Before:** Ridge Regression trained on 4 Texas gubernatorial races (2010-2022)
**After:** Hierarchical Bayesian model trained on 500+ races (Governor, Senate, House 2010-2024)

---

## File Changes Summary

### Files to KEEP (no changes needed)
| File | Reason |
|------|--------|
| `requirements.txt` | Dependencies still needed |
| `pyproject.toml` | Project config unchanged |
| `visualizations.py` | Visualization utilities are race-agnostic |
| `notifications.py` | Alert system is race-agnostic |
| VIX/Macro data loaders | Economic data is universal |

### Files to MODIFY
| File | Changes |
|------|---------|
| `Model.py` | Replace `PredictiveModel` class (lines 2141-2690) with import from `models/hierarchical_model.py` |
| `database.py` | Replace `TEXAS_GOVERNOR` schema with `ELECTION_PREDICTIONS` (use `database_v2.py`) |
| `clean.py` | Replace Texas-specific loaders (lines 5027-6443) with imports from `data_loader_v2.py` |
| `ETL.py` | Update pipeline to use parameterized config instead of hardcoded years |
| `streamlit_app.py` | Add race selector, update to use new prediction format |

### Files to ADD
| File | Purpose |
|------|---------|
| `config/race_config.py` | Race configuration system |
| `models/hierarchical_model.py` | Hierarchical Bayesian/mixed-effects model |
| `database_v2.py` | Multi-race Snowflake schema |
| `data_loader_v2.py` | Generalized data loading |

---

## Feature Hierarchy

### Tier 1: Universal (Always Available)
```python
TIER1_FEATURES = [
    'partisan_lean',           # State Cook PVI
    'incumbency',              # -1, 0, or 1
    'funding_ratio',           # R raised / D raised
    'gdp_growth',              # Latest quarterly
    'unemployment_rate',       # Current
    'inflation_rate',          # Current CPI
    'national_environment',    # Generic ballot
    'presidential_approval',   # Current approval
    'election_context',        # presidential/midterm
    'same_party_as_president', # Binary
]
```

### Tier 2: Enhanced (When Available)
```python
TIER2_FEATURES = [
    'poll_margin_mean',        # Polling average
    'poll_margin_std',         # Polling volatility
    'poll_count',              # Number of polls
    'vix_mean',                # Market volatility
    'news_sentiment_positive', # FinBERT sentiment
    'news_sentiment_negative', # FinBERT sentiment
]
```

### Tier 3: Race-Specific (Situational)
```python
TIER3_FEATURES = [
    'midterm_penalty',         # In-party midterm effect
    'coattail_effect',         # Presidential coattails
    'culture_war_exposure',    # Culture war events
    'candidate_quality_r',     # R candidate quality score
    'candidate_quality_d',     # D candidate quality score
]
```

---

## Model Architecture Changes

### Before (Model.py:2540-2643)
```python
# Single Ridge regression, no hierarchy
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)
# No confidence intervals
# No partial pooling
# n=4 observations
```

### After (models/hierarchical_model.py)
```python
# Hierarchical Bayesian model
# margin_ij = β₀ + β₁·X_ij + α_racetype[j] + α_state[j] + α_year[j] + ε_ij

with pm.Model():
    # Hyperpriors for partial pooling
    sigma_race = pm.HalfNormal('sigma_race', sigma=5)
    sigma_state = pm.HalfNormal('sigma_state', sigma=5)

    # Random effects
    alpha_race = pm.Normal('alpha_race', mu=0, sigma=sigma_race, shape=n_race_types)
    alpha_state = pm.Normal('alpha_state', mu=0, sigma=sigma_state, shape=n_states)

    # Fixed effects
    beta = pm.Normal('beta', mu=0, sigma=5, shape=n_features)

    # Full posterior for uncertainty quantification
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

**Benefits:**
- Partial pooling: States with sparse data borrow from similar states
- Proper uncertainty: 95% confidence intervals from posterior
- Low-confidence flags: Automatic flagging when CI > 20%

---

## Database Schema Changes

### Before (database.py)
```sql
-- Schema: TEXAS_GOVERNOR
-- Tables hardcoded for Texas Governor only

CREATE TABLE ELECTION_RESULTS_STATEWIDE (
    ELECTION_YEAR INTEGER,
    STATE VARCHAR(2),  -- Always 'TX'
    RACE VARCHAR(50),  -- Always 'Governor'
    ...
);
```

### After (database_v2.py)
```sql
-- Schema: ELECTION_PREDICTIONS
-- Multi-race support with dimension tables

-- Dimension table for race types
CREATE TABLE DIM_RACE_TYPE (
    RACE_TYPE_ID INTEGER,
    RACE_TYPE VARCHAR(50),   -- governor, us_senate, us_house
    RACE_LEVEL VARCHAR(20),  -- federal, state
    CYCLE_LENGTH INTEGER,    -- 2, 4, or 6
    PRIMARY KEY (RACE_TYPE_ID)
);

-- Unified results table
CREATE TABLE ELECTION_RESULTS (
    RESULT_ID INTEGER AUTOINCREMENT,
    ELECTION_YEAR INTEGER,
    RACE_TYPE VARCHAR(50),   -- NEW: governor, us_senate, us_house
    STATE VARCHAR(2),
    DISTRICT INTEGER,         -- NEW: For House races
    MARGIN_PCT FLOAT,         -- NEW: Pre-calculated margin
    ...
);

-- NEW: Model predictions with confidence
CREATE TABLE MODEL_PREDICTIONS (
    PREDICTION_ID INTEGER AUTOINCREMENT,
    MODEL_VERSION VARCHAR(50),
    RACE_TYPE VARCHAR(50),
    STATE VARCHAR(2),
    MARGIN_ESTIMATE FLOAT,
    MARGIN_CI_LOWER_95 FLOAT,  -- NEW: 95% CI
    MARGIN_CI_UPPER_95 FLOAT,
    WIN_PROB_R FLOAT,          -- NEW: Win probability
    IS_LOW_CONFIDENCE BOOLEAN, -- NEW: Flag for sparse data
    ...
);
```

---

## Code Migration Steps

### Step 1: Set up new directory structure
```bash
mkdir -p config models
mv config/race_config.py config/
mv models/hierarchical_model.py models/
```

### Step 2: Update imports in Model.py
```python
# REMOVE these lines (2141-2690):
# class PredictiveModel:
#     ...

# ADD this import at top:
from models.hierarchical_model import (
    HierarchicalElectionModel,
    FeatureEngineer,
    predict_election,
    PredictionResult
)
```

### Step 3: Update data loading in clean.py
```python
# REMOVE Texas-specific functions (lines 5027-6443):
# def load_texas_governor_election_data()
# def load_texas_campaign_finance_data()
# def load_texas_governor_polling_data()

# ADD this import at top:
from data_loader_v2 import (
    load_race_data,
    ElectionResultsLoader,
    CampaignFinanceLoader,
    PollingDataLoader,
    HistoricalDataAggregator
)
```

### Step 4: Update ETL.py for multi-race
```python
# MODIFY ETLPipeline class:

class ETLPipeline:
    def __init__(
        self,
        race_configs: List[RaceConfig],  # NEW: Accept config list
        start_year: int = 2010,
        end_year: int = 2025
    ):
        self.race_configs = race_configs
        # ...

    def run(self):
        for config in self.race_configs:
            data = load_race_data(
                state=config.state,
                race_type=config.race_type.value,
                election_years=config.election_years
            )
            # Process each race...
```

### Step 5: Run Snowflake migration
```bash
# Generate migration SQL
python database_v2.py --migrate > migration.sql

# Review and execute
snowsql -f migration.sql
```

---

## Prediction Interface

### Before
```python
# Hardcoded prediction
model = PredictiveModel(manager)
model.prepare_model_data()
results = model.run_logistic_regression()
# Returns point estimate only
```

### After
```python
from models.hierarchical_model import predict_election, HierarchicalElectionModel

# Train on historical data
model = HierarchicalElectionModel(use_bayesian=True)
model.fit(X_train, y_train, race_types, states, years)

# Predict any race
result = predict_election(
    state='TX',
    race_type='governor',
    election_year=2026,
    model=model,
    macro_data={'gdp_growth': 2.5, 'unemployment_rate': 4.0, ...},
    polling_data={'margin_mean': 8.5, 'margin_std': 3.2},
    incumbent_party='R'
)

# Full result with uncertainty
print(f"Margin: {result.point_estimate:.1f}%")
print(f"95% CI: [{result.lower_95:.1f}, {result.upper_95:.1f}]")
print(f"R win prob: {result.win_probability_r:.1%}")

if result.is_low_confidence:
    print("⚠️ LOW CONFIDENCE: Sparse data for this race")
```

---

## Texas-Specific Data to Remove

### Hardcoded in clean.py (REMOVE)

| Location | Content | Action |
|----------|---------|--------|
| Lines 5077-5121 | `governor_elections` dict with Rick Perry, Greg Abbott | Move to JSON config |
| Lines 5137-5173 | `official_results` vote counts | Move to CSV/database |
| Lines 5486-5543 | `governor_candidates` filer IDs | Move to config |
| Lines 6247-6443 | Hardcoded poll records | Load from 538 API |

### Hardcoded in Model.py (REMOVE)

| Location | Content | Action |
|----------|---------|--------|
| Line 2170 | `election_years = [2010, 2014, 2018, 2022]` | Use RaceConfig |
| Line 2269 | `news_path = './data/news/texas_governor_news.csv'` | Parameterize |

### Hardcoded in database.py (REMOVE)

| Location | Content | Action |
|----------|---------|--------|
| Line 69 | `DEFAULT_SCHEMA = 'TEXAS_GOVERNOR'` | Change to `ELECTION_PREDICTIONS` |
| Lines 75-100 | Docstring mentioning Texas tables | Update documentation |

---

## Testing the Refactored Pipeline

```python
# test_refactored_pipeline.py

from config.race_config import RaceConfig, RaceType
from models.hierarchical_model import (
    HierarchicalElectionModel,
    FeatureEngineer,
    predict_election
)
from data_loader_v2 import HistoricalDataAggregator

# 1. Build training data
aggregator = HistoricalDataAggregator(years=list(range(2010, 2023, 2)))
training_df = aggregator.build_training_dataset()
print(f"Training samples: {len(training_df)}")

# 2. Prepare features
engineer = FeatureEngineer()
X, y = engineer.build_feature_matrix(feature_sets)

# 3. Train model
model = HierarchicalElectionModel(use_bayesian=False)  # Start with frequentist
model.fit(X, y,
          race_types=training_df['race_type'].values,
          states=training_df['state'].values,
          years=training_df['election_year'].values)

# 4. Validate on Texas 2022 (should match known result)
result = predict_election(
    state='TX',
    race_type='governor',
    election_year=2022,
    model=model,
    macro_data={'gdp_growth': 2.9, 'unemployment_rate': 4.2, 'inflation_rate': 8.0},
    polling_data={'margin_mean': 9.0, 'margin_std': 4.5},
    incumbent_party='R'
)

print(f"Predicted: R+{result.point_estimate:.1f}%")
print(f"Actual: R+11.1%")
print(f"Error: {abs(result.point_estimate - 11.1):.1f}%")
```

---

## Dependencies to Add

```
# requirements.txt additions
pymc>=5.0.0        # Bayesian modeling (optional but recommended)
arviz>=0.15.0      # Posterior analysis
statsmodels>=0.14  # Mixed-effects models (fallback)
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Training data | n=4 (TX Gov 2010-2022) | n=500+ (multi-race) |
| Model | Ridge regression | Hierarchical Bayesian |
| Output | Point estimate | Margin + 95% CI + win prob |
| Uncertainty | None | Full posterior |
| Low-confidence flag | No | Yes |
| Race types | Texas Governor only | Governor, Senate, House |
| States | Texas only | All 50 states |
| Schema | TEXAS_GOVERNOR | ELECTION_PREDICTIONS |
| Configuration | Hardcoded | JSON-based RaceConfig |
