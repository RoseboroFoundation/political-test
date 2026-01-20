"""
Hierarchical Election Prediction Model

This module implements a hierarchical Bayesian model for election prediction
that generalizes across race types, states, and years.

REFACTORING NOTES:
- [ADD] New file - replaces single Ridge regression in Model.py:2540-2643
- [ADD] Mixed-effects structure: fixed effects + random effects by race_type, state
- [ADD] Handles variable feature availability (sparse data)
- [ADD] Outputs margin prediction + confidence intervals
- [ADD] Supports 500+ training observations instead of n=4

Model Architecture:
    margin_ij = β₀ + β₁·X_ij + α_racetype[j] + α_state[j] + α_year[j] + ε_ij

Where:
    - β₀, β₁ are fixed effects (universal features)
    - α_racetype[j] is race-type random effect (Governor vs Senate vs House)
    - α_state[j] is state random effect (captures state-level idiosyncrasies)
    - α_year[j] is year random effect (captures wave years, national environment)
    - ε_ij is residual error
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
import warnings

# ML/Stats imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import Bayesian libraries (optional but recommended)
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Fallback to frequentist mixed effects
try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    MIXED_LM_AVAILABLE = True
except ImportError:
    MIXED_LM_AVAILABLE = False

# Config imports
import sys
sys.path.insert(0, '..')
from config.race_config import (
    RaceConfig, RaceType, ElectionContext, ElectionResult,
    FEATURE_TIERS, STATE_PARTISAN_LEAN
)

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE ENGINEERING (GENERALIZABLE)
# =============================================================================

@dataclass
class FeatureSet:
    """
    Container for election features with availability tracking.

    Replaces hardcoded feature extraction in Model.py:2160-2248
    with a flexible system that handles missing data.
    """
    election_year: int
    state: str
    race_type: str
    district: Optional[int] = None

    # Target variable
    margin_pct: Optional[float] = None  # R margin (positive = R win)

    # Tier 1: Universal features (always required)
    partisan_lean: float = 0.0
    incumbency: int = 0  # 1 if R incumbent, -1 if D incumbent, 0 if open
    funding_ratio: float = 1.0
    gdp_growth: float = 2.0
    unemployment_rate: float = 5.0
    inflation_rate: float = 2.5
    national_environment: float = 0.0  # Generic ballot
    presidential_approval: float = 45.0
    election_context: str = 'midterm'
    same_party_as_president: int = 0

    # Tier 2: Enhanced features (optional)
    poll_margin_mean: Optional[float] = None
    poll_margin_std: Optional[float] = None
    poll_count: Optional[int] = None
    vix_mean: Optional[float] = None
    news_sentiment_positive: Optional[float] = None
    news_sentiment_negative: Optional[float] = None

    # Tier 3: Race-specific (optional)
    midterm_penalty: Optional[float] = None
    coattail_effect: Optional[float] = None
    culture_war_exposure: Optional[int] = None
    candidate_quality_r: Optional[float] = None
    candidate_quality_d: Optional[float] = None

    # Metadata
    has_polling: bool = False
    has_sentiment: bool = False
    confidence_tier: int = 1  # 1=low, 2=medium, 3=high

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame construction."""
        return {k: v for k, v in self.__dict__.items()}

    @property
    def feature_completeness(self) -> float:
        """Calculate what percentage of features are available."""
        total_features = 20
        available = sum([
            self.poll_margin_mean is not None,
            self.poll_margin_std is not None,
            self.vix_mean is not None,
            self.news_sentiment_positive is not None,
            self.midterm_penalty is not None,
            self.coattail_effect is not None,
            self.culture_war_exposure is not None,
        ])
        # Tier 1 always counts as full
        return (10 + available) / total_features


class FeatureEngineer:
    """
    Generalizable feature engineering for any election race.

    REPLACES: Model.py PredictiveModel.prepare_model_data() (lines 2160-2248)

    Key changes:
    - Parameterized by RaceConfig instead of hardcoded Texas years
    - Handles missing features gracefully
    - Supports multiple race types with different feature availability
    """

    def __init__(self, config: Optional[RaceConfig] = None):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def extract_features(
        self,
        election_result: ElectionResult,
        macro_data: Dict[str, float],
        polling_data: Optional[Dict] = None,
        finance_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> FeatureSet:
        """
        Extract features for a single election.

        Args:
            election_result: Election outcome data
            macro_data: Macroeconomic indicators
            polling_data: Polling aggregates (optional)
            finance_data: Campaign finance data (optional)
            news_data: News/sentiment data (optional)
            market_data: VIX/market data (optional)

        Returns:
            FeatureSet with all available features
        """
        fs = FeatureSet(
            election_year=election_result.year,
            state=election_result.state,
            race_type=election_result.race_type.value,
            district=election_result.district,
            margin_pct=election_result.margin_pct
        )

        # Tier 1: Universal features
        fs.partisan_lean = STATE_PARTISAN_LEAN.get(election_result.state, 0.0)

        # Determine incumbency
        r_incumbent = election_result.candidates.get('R', {}).get('incumbent', False)
        d_incumbent = election_result.candidates.get('D', {}).get('incumbent', False)
        if r_incumbent:
            fs.incumbency = 1
        elif d_incumbent:
            fs.incumbency = -1
        else:
            fs.incumbency = 0

        # Campaign finance
        if finance_data:
            r_raised = finance_data.get('r_raised', 1)
            d_raised = finance_data.get('d_raised', 1)
            fs.funding_ratio = r_raised / max(d_raised, 1)

        # Macroeconomic
        fs.gdp_growth = macro_data.get('gdp_growth', 2.0)
        fs.unemployment_rate = macro_data.get('unemployment_rate', 5.0)
        fs.inflation_rate = macro_data.get('inflation_rate', 2.5)
        fs.national_environment = macro_data.get('generic_ballot', 0.0)
        fs.presidential_approval = macro_data.get('presidential_approval', 45.0)

        # Election context
        if election_result.year % 4 == 0:
            fs.election_context = 'presidential'
        elif election_result.year % 2 == 0:
            fs.election_context = 'midterm'
        else:
            fs.election_context = 'off_year'

        # Same party as president calculation
        pres_party = macro_data.get('president_party', 'D')
        incumbent_party = 'R' if fs.incumbency == 1 else ('D' if fs.incumbency == -1 else None)
        fs.same_party_as_president = 1 if incumbent_party == pres_party else 0

        # Tier 2: Enhanced features
        if polling_data:
            fs.poll_margin_mean = polling_data.get('margin_mean')
            fs.poll_margin_std = polling_data.get('margin_std')
            fs.poll_count = polling_data.get('count')
            fs.has_polling = fs.poll_margin_mean is not None

        if market_data:
            fs.vix_mean = market_data.get('vix_mean')

        if news_data:
            fs.news_sentiment_positive = news_data.get('sentiment_positive')
            fs.news_sentiment_negative = news_data.get('sentiment_negative')
            fs.has_sentiment = fs.news_sentiment_positive is not None

        # Tier 3: Race-specific adjustments
        if fs.election_context == 'midterm' and fs.same_party_as_president:
            fs.midterm_penalty = -3.0  # Historical average penalty
        else:
            fs.midterm_penalty = 0.0

        if fs.election_context == 'presidential':
            # Coattail effect based on presidential margin
            pres_margin = macro_data.get('presidential_margin', 0)
            fs.coattail_effect = pres_margin * 0.3  # 30% coattail

        # Culture war events (if available)
        if news_data and 'culture_war_events' in news_data:
            fs.culture_war_exposure = news_data['culture_war_events']

        # Set confidence tier based on data availability
        if fs.has_polling and fs.has_sentiment:
            fs.confidence_tier = 3
        elif fs.has_polling or fs.vix_mean is not None:
            fs.confidence_tier = 2
        else:
            fs.confidence_tier = 1

        return fs

    def build_feature_matrix(
        self,
        feature_sets: List[FeatureSet],
        include_tier2: bool = True,
        include_tier3: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build feature matrix for modeling.

        Handles missing features via imputation or exclusion.

        Returns:
            Tuple of (X feature DataFrame, y target array)
        """
        records = [fs.to_dict() for fs in feature_sets]
        df = pd.DataFrame(records)

        # Define feature columns by tier
        tier1_cols = [
            'partisan_lean', 'incumbency', 'funding_ratio',
            'gdp_growth', 'unemployment_rate', 'inflation_rate',
            'national_environment', 'presidential_approval',
            'same_party_as_president'
        ]

        tier2_cols = [
            'poll_margin_mean', 'poll_margin_std', 'vix_mean',
            'news_sentiment_positive', 'news_sentiment_negative'
        ]

        tier3_cols = [
            'midterm_penalty', 'coattail_effect', 'culture_war_exposure'
        ]

        # Select features based on tiers
        feature_cols = tier1_cols.copy()
        if include_tier2:
            feature_cols.extend(tier2_cols)
        if include_tier3:
            feature_cols.extend(tier3_cols)

        # Filter to existing columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].copy()
        y = df['margin_pct'].values

        # Impute missing values
        # For Tier 2/3, use median imputation or indicator variables
        for col in X.columns:
            if X[col].isna().any():
                # Create missingness indicator
                X[f'{col}_missing'] = X[col].isna().astype(int)
                # Impute with median
                X[col] = X[col].fillna(X[col].median())

        # Encode categorical: election_context
        if 'election_context' in df.columns:
            le = LabelEncoder()
            X['election_context_encoded'] = le.fit_transform(df['election_context'])
            self.label_encoders['election_context'] = le

        # Add group indicators for hierarchical model
        X['race_type'] = df['race_type']
        X['state'] = df['state']
        X['year'] = df['election_year']

        return X, y


# =============================================================================
# HIERARCHICAL MODEL
# =============================================================================

@dataclass
class PredictionResult:
    """
    Result of a single prediction with uncertainty quantification.

    REPLACES: Simple point predictions in Model.py
    ADDS: Confidence intervals, low-confidence flags
    """
    point_estimate: float           # Predicted R margin
    lower_95: float                 # 95% CI lower bound
    upper_95: float                 # 95% CI upper bound
    lower_80: float                 # 80% CI lower bound
    upper_80: float                 # 80% CI upper bound
    std_error: float                # Standard error
    win_probability_r: float        # P(R wins)
    win_probability_d: float        # P(D wins)
    confidence_tier: int            # 1=low, 2=medium, 3=high
    is_low_confidence: bool         # Flag for predictions with high uncertainty
    feature_completeness: float     # Fraction of features available
    contributing_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'margin_estimate': round(self.point_estimate, 2),
            'margin_ci_95': [round(self.lower_95, 2), round(self.upper_95, 2)],
            'margin_ci_80': [round(self.lower_80, 2), round(self.upper_80, 2)],
            'std_error': round(self.std_error, 2),
            'win_prob_r': round(self.win_probability_r, 3),
            'win_prob_d': round(self.win_probability_d, 3),
            'confidence_tier': self.confidence_tier,
            'is_low_confidence': self.is_low_confidence,
            'feature_completeness': round(self.feature_completeness, 2),
            'top_factors': dict(sorted(
                self.contributing_factors.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5])
        }


class HierarchicalElectionModel:
    """
    Hierarchical Bayesian model for election prediction.

    REPLACES: Ridge regression in Model.py:2540-2643

    Key improvements:
    1. Partial pooling across race types and states
    2. Proper uncertainty quantification
    3. Handles small samples by borrowing strength
    4. Explicit modeling of year effects (wave elections)
    5. Confidence flags for sparse-data predictions
    """

    def __init__(
        self,
        use_bayesian: bool = True,
        n_samples: int = 2000,
        n_chains: int = 4
    ):
        """
        Initialize hierarchical model.

        Args:
            use_bayesian: If True, use PyMC for full Bayesian inference.
                         If False, use frequentist MixedLM.
            n_samples: Number of MCMC samples (Bayesian only)
            n_chains: Number of MCMC chains (Bayesian only)
        """
        self.use_bayesian = use_bayesian and BAYESIAN_AVAILABLE
        self.n_samples = n_samples
        self.n_chains = n_chains

        self.is_fitted = False
        self.trace = None
        self.mixed_model = None
        self.scaler = StandardScaler()
        self.feature_cols = []

        # Model diagnostics
        self.posterior_predictive = None
        self.loo_score = None
        self.rmse = None
        self.mae = None

        # Random effect estimates
        self.race_type_effects = {}
        self.state_effects = {}
        self.year_effects = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        race_types: np.ndarray,
        states: np.ndarray,
        years: np.ndarray
    ) -> 'HierarchicalElectionModel':
        """
        Fit the hierarchical model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target margin percentages
            race_types: Array of race type labels
            states: Array of state codes
            years: Array of election years

        Returns:
            self (fitted model)
        """
        logger.info(f"Fitting hierarchical model on {len(y)} observations...")

        # Identify numeric feature columns (exclude grouping variables)
        self.feature_cols = [c for c in X.columns
                           if c not in ['race_type', 'state', 'year']
                           and X[c].dtype in ['int64', 'float64']]

        X_numeric = X[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X_numeric)

        if self.use_bayesian:
            return self._fit_bayesian(X_scaled, y, race_types, states, years)
        else:
            return self._fit_frequentist(X, y, race_types, states, years)

    def _fit_bayesian(
        self,
        X: np.ndarray,
        y: np.ndarray,
        race_types: np.ndarray,
        states: np.ndarray,
        years: np.ndarray
    ) -> 'HierarchicalElectionModel':
        """Fit Bayesian hierarchical model using PyMC."""
        logger.info("Fitting Bayesian hierarchical model...")

        # Encode categorical variables
        race_type_encoder = LabelEncoder()
        state_encoder = LabelEncoder()
        year_encoder = LabelEncoder()

        race_type_idx = race_type_encoder.fit_transform(race_types)
        state_idx = state_encoder.fit_transform(states)
        year_idx = year_encoder.fit_transform(years)

        n_race_types = len(race_type_encoder.classes_)
        n_states = len(state_encoder.classes_)
        n_years = len(year_encoder.classes_)
        n_features = X.shape[1]

        with pm.Model() as model:
            # Hyperpriors
            sigma_race = pm.HalfNormal('sigma_race', sigma=5)
            sigma_state = pm.HalfNormal('sigma_state', sigma=5)
            sigma_year = pm.HalfNormal('sigma_year', sigma=5)

            # Fixed effects (feature coefficients)
            beta = pm.Normal('beta', mu=0, sigma=5, shape=n_features)
            intercept = pm.Normal('intercept', mu=0, sigma=10)

            # Random effects
            alpha_race = pm.Normal('alpha_race', mu=0, sigma=sigma_race, shape=n_race_types)
            alpha_state = pm.Normal('alpha_state', mu=0, sigma=sigma_state, shape=n_states)
            alpha_year = pm.Normal('alpha_year', mu=0, sigma=sigma_year, shape=n_years)

            # Model error
            sigma = pm.HalfNormal('sigma', sigma=10)

            # Linear predictor
            mu = (
                intercept
                + pm.math.dot(X, beta)
                + alpha_race[race_type_idx]
                + alpha_state[state_idx]
                + alpha_year[year_idx]
            )

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            # Sample
            self.trace = pm.sample(
                self.n_samples,
                chains=self.n_chains,
                return_inferencedata=True,
                progressbar=True
            )

            # Posterior predictive
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                extend_inferencedata=True
            )

        # Extract random effect means
        self.race_type_effects = dict(zip(
            race_type_encoder.classes_,
            self.trace.posterior['alpha_race'].mean(dim=['chain', 'draw']).values
        ))
        self.state_effects = dict(zip(
            state_encoder.classes_,
            self.trace.posterior['alpha_state'].mean(dim=['chain', 'draw']).values
        ))
        self.year_effects = dict(zip(
            year_encoder.classes_,
            self.trace.posterior['alpha_year'].mean(dim=['chain', 'draw']).values
        ))

        # Store encoders
        self._race_type_encoder = race_type_encoder
        self._state_encoder = state_encoder
        self._year_encoder = year_encoder

        # Calculate fit metrics
        y_pred = self.trace.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))
        self.mae = mean_absolute_error(y, y_pred)

        # LOO cross-validation score
        self.loo_score = az.loo(self.trace)

        logger.info(f"Bayesian model fitted. RMSE: {self.rmse:.2f}, MAE: {self.mae:.2f}")

        self.is_fitted = True
        return self

    def _fit_frequentist(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        race_types: np.ndarray,
        states: np.ndarray,
        years: np.ndarray
    ) -> 'HierarchicalElectionModel':
        """Fit frequentist mixed-effects model using statsmodels."""
        logger.info("Fitting frequentist mixed-effects model...")

        if not MIXED_LM_AVAILABLE:
            raise ImportError("statsmodels required for frequentist mixed model")

        # Prepare data
        df = X.copy()
        df['y'] = y
        df['race_type'] = race_types
        df['state'] = states
        df['year'] = years

        # Create formula
        # Fixed effects: all numeric features
        fixed_formula = ' + '.join(self.feature_cols)
        formula = f"y ~ {fixed_formula}"

        # Fit with nested random effects
        # Random intercepts for race_type, state, and year
        model = MixedLM.from_formula(
            formula,
            data=df,
            groups=df['state'],
            re_formula='1',
            vc_formula={'race_type': '0 + C(race_type)', 'year': '0 + C(year)'}
        )

        self.mixed_model = model.fit(method='lbfgs')

        # Extract random effects
        self.state_effects = self.mixed_model.random_effects

        # Calculate fit metrics
        y_pred = self.mixed_model.fittedvalues
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))
        self.mae = mean_absolute_error(y, y_pred)

        logger.info(f"Mixed-effects model fitted. RMSE: {self.rmse:.2f}, MAE: {self.mae:.2f}")

        self.is_fitted = True
        return self

    def predict(
        self,
        feature_set: FeatureSet,
        return_samples: bool = False
    ) -> PredictionResult:
        """
        Make prediction with uncertainty quantification.

        Args:
            feature_set: Features for prediction
            return_samples: If True, also return posterior samples

        Returns:
            PredictionResult with point estimate and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Build feature vector
        features = {col: getattr(feature_set, col, 0)
                   for col in self.feature_cols}
        X = np.array([[features.get(c, 0) for c in self.feature_cols]])
        X_scaled = self.scaler.transform(X)

        if self.use_bayesian:
            return self._predict_bayesian(X_scaled, feature_set, return_samples)
        else:
            return self._predict_frequentist(X_scaled, feature_set)

    def _predict_bayesian(
        self,
        X: np.ndarray,
        feature_set: FeatureSet,
        return_samples: bool
    ) -> PredictionResult:
        """Make Bayesian prediction with full posterior."""
        # Get posterior samples of coefficients
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, len(self.feature_cols))
        intercept_samples = self.trace.posterior['intercept'].values.flatten()
        sigma_samples = self.trace.posterior['sigma'].values.flatten()

        # Get random effects for this observation
        race_effect = self.race_type_effects.get(feature_set.race_type, 0)
        state_effect = self.state_effects.get(feature_set.state, 0)
        year_effect = self.year_effects.get(feature_set.election_year, 0)

        # Calculate posterior predictive samples
        mu_samples = (
            intercept_samples
            + np.dot(beta_samples, X.flatten())
            + race_effect
            + state_effect
            + year_effect
        )

        # Add observation noise
        pred_samples = mu_samples + np.random.normal(0, sigma_samples)

        # Calculate statistics
        point_estimate = np.mean(pred_samples)
        std_error = np.std(pred_samples)

        # Confidence intervals
        lower_95, upper_95 = np.percentile(pred_samples, [2.5, 97.5])
        lower_80, upper_80 = np.percentile(pred_samples, [10, 90])

        # Win probability (R wins if margin > 0)
        win_prob_r = np.mean(pred_samples > 0)
        win_prob_d = 1 - win_prob_r

        # Calculate feature contributions
        beta_mean = beta_samples.mean(axis=0)
        contributions = dict(zip(
            self.feature_cols,
            (beta_mean * X.flatten()).tolist()
        ))

        # Determine if low confidence
        is_low_confidence = (
            (upper_95 - lower_95) > 20  # Wide CI
            or feature_set.confidence_tier == 1
            or feature_set.feature_completeness < 0.5
        )

        return PredictionResult(
            point_estimate=point_estimate,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_80=lower_80,
            upper_80=upper_80,
            std_error=std_error,
            win_probability_r=win_prob_r,
            win_probability_d=win_prob_d,
            confidence_tier=feature_set.confidence_tier,
            is_low_confidence=is_low_confidence,
            feature_completeness=feature_set.feature_completeness,
            contributing_factors=contributions
        )

    def _predict_frequentist(
        self,
        X: np.ndarray,
        feature_set: FeatureSet
    ) -> PredictionResult:
        """Make frequentist prediction with bootstrap CIs."""
        # Point prediction
        point_estimate = self.mixed_model.predict(
            exog=dict(zip(self.feature_cols, X.flatten()))
        ).values[0]

        # Bootstrap for CI (simplified)
        std_error = self.mixed_model.bse_fe.mean() * 2  # Rough approximation

        lower_95 = point_estimate - 1.96 * std_error
        upper_95 = point_estimate + 1.96 * std_error
        lower_80 = point_estimate - 1.28 * std_error
        upper_80 = point_estimate + 1.28 * std_error

        # Win probability (normal approximation)
        from scipy import stats
        win_prob_r = 1 - stats.norm.cdf(0, point_estimate, std_error)
        win_prob_d = 1 - win_prob_r

        is_low_confidence = (
            (upper_95 - lower_95) > 20
            or feature_set.confidence_tier == 1
        )

        return PredictionResult(
            point_estimate=point_estimate,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_80=lower_80,
            upper_80=upper_80,
            std_error=std_error,
            win_probability_r=win_prob_r,
            win_probability_d=win_prob_d,
            confidence_tier=feature_set.confidence_tier,
            is_low_confidence=is_low_confidence,
            feature_completeness=feature_set.feature_completeness,
            contributing_factors={}
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics and summaries."""
        diagnostics = {
            'is_fitted': self.is_fitted,
            'model_type': 'Bayesian' if self.use_bayesian else 'Frequentist',
            'rmse': self.rmse,
            'mae': self.mae,
            'race_type_effects': self.race_type_effects,
            'state_effects': dict(list(self.state_effects.items())[:10]),  # Top 10
            'year_effects': self.year_effects
        }

        if self.use_bayesian and self.trace is not None:
            diagnostics['loo_score'] = float(self.loo_score.elpd_loo) if self.loo_score else None
            diagnostics['n_samples'] = self.n_samples
            diagnostics['n_chains'] = self.n_chains

        return diagnostics


# =============================================================================
# PREDICTION FUNCTION (MAIN INTERFACE)
# =============================================================================

def predict_election(
    state: str,
    race_type: str,
    election_year: int,
    model: HierarchicalElectionModel,
    macro_data: Dict,
    polling_data: Optional[Dict] = None,
    finance_data: Optional[Dict] = None,
    news_data: Optional[Dict] = None,
    market_data: Optional[Dict] = None,
    incumbent_party: Optional[str] = None,
    district: Optional[int] = None
) -> PredictionResult:
    """
    Main prediction interface.

    REPLACES: Model.py _run_margin_regression() (lines 2540-2643)

    This is the primary function to call for making predictions.
    Handles variable feature availability and flags low-confidence predictions.

    Args:
        state: Two-letter state code
        race_type: 'governor', 'us_senate', or 'us_house'
        election_year: Year of election
        model: Fitted HierarchicalElectionModel
        macro_data: Dict with gdp_growth, unemployment_rate, inflation_rate,
                   generic_ballot, presidential_approval, president_party
        polling_data: Optional dict with margin_mean, margin_std, count
        finance_data: Optional dict with r_raised, d_raised
        news_data: Optional dict with sentiment_positive, sentiment_negative
        market_data: Optional dict with vix_mean
        incumbent_party: 'R', 'D', or None for open seat
        district: Congressional district (US House only)

    Returns:
        PredictionResult with margin estimate, confidence intervals, and flags

    Example:
        >>> model = HierarchicalElectionModel()
        >>> model.fit(X_train, y_train, race_types, states, years)
        >>> result = predict_election(
        ...     state='TX',
        ...     race_type='governor',
        ...     election_year=2026,
        ...     model=model,
        ...     macro_data={'gdp_growth': 2.5, 'unemployment_rate': 4.0, ...},
        ...     polling_data={'margin_mean': 8.5, 'margin_std': 3.2, 'count': 12}
        ... )
        >>> print(f"Predicted R margin: {result.point_estimate:.1f}%")
        >>> print(f"95% CI: [{result.lower_95:.1f}, {result.upper_95:.1f}]")
        >>> if result.is_low_confidence:
        ...     print("WARNING: Low confidence prediction")
    """
    # Create feature engineer
    engineer = FeatureEngineer()

    # Build mock ElectionResult for feature extraction
    mock_result = ElectionResult(
        year=election_year,
        state=state,
        race_type=RaceType(race_type),
        district=district,
        winner='TBD',
        winner_party='TBD',
        margin_pct=0,  # Unknown
        total_votes=0,
        turnout_pct=0,
        candidates={
            'R': {'name': 'Republican', 'incumbent': incumbent_party == 'R'},
            'D': {'name': 'Democrat', 'incumbent': incumbent_party == 'D'}
        },
        election_date=f'{election_year}-11-01'
    )

    # Extract features
    features = engineer.extract_features(
        election_result=mock_result,
        macro_data=macro_data,
        polling_data=polling_data,
        finance_data=finance_data,
        news_data=news_data,
        market_data=market_data
    )

    # Make prediction
    result = model.predict(features)

    return result


# =============================================================================
# BATCH PREDICTION
# =============================================================================

def predict_batch(
    races: List[Dict],
    model: HierarchicalElectionModel,
    macro_data: Dict
) -> List[Tuple[Dict, PredictionResult]]:
    """
    Make predictions for multiple races.

    Args:
        races: List of race configurations
        model: Fitted model
        macro_data: Macroeconomic data

    Returns:
        List of (race_config, prediction) tuples
    """
    results = []

    for race in races:
        try:
            prediction = predict_election(
                state=race['state'],
                race_type=race['race_type'],
                election_year=race['year'],
                model=model,
                macro_data=macro_data,
                polling_data=race.get('polling'),
                finance_data=race.get('finance'),
                incumbent_party=race.get('incumbent_party')
            )
            results.append((race, prediction))
        except Exception as e:
            logger.warning(f"Failed to predict {race}: {e}")
            continue

    return results
