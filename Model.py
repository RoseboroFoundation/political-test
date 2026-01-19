"""
Statistical Model and Descriptive Analysis for Texas Governor Race Data.

This module provides descriptive statistics and analytical models using
data from the Snowflake database. It includes summary statistics,
distributions, correlations, and trend analysis.

Usage:
    python Model.py --summary              # Run all summary statistics
    python Model.py --elections            # Election statistics
    python Model.py --finance              # Campaign finance statistics
    python Model.py --polling              # Polling statistics
    python Model.py --news                 # News coverage statistics
    python Model.py --correlations         # Cross-dataset correlations
    python Model.py --export               # Export results to files
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

import pandas as pd
import numpy as np

# Import from project modules
from database import (
    DatabaseManager,
    SnowflakeConnection,
    SNOWFLAKE_TABLES,
    DEFAULT_SCHEMA
)
from ETL import (
    ETLPipeline,
    DATA_DICTIONARY,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR
)

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory for reports
OUTPUT_DIR = './output/statistics'


# =============================================================================
# DESCRIPTIVE STATISTICS BASE CLASS
# =============================================================================
class DescriptiveStatistics:
    """
    Base class for computing descriptive statistics.
    """

    @staticmethod
    def convert_decimals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Decimal columns to float for compatibility with pandas operations.
        Snowflake returns numeric columns as Decimal objects.

        Args:
            df: DataFrame potentially containing Decimal columns

        Returns:
            DataFrame with Decimal columns converted to float
        """
        if df is None or df.empty:
            return df

        from decimal import Decimal
        for col in df.columns:
            if df[col].dtype == object:
                # Check if column contains Decimal objects
                sample = df[col].dropna().head(1)
                if len(sample) > 0 and isinstance(sample.iloc[0], Decimal):
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            elif hasattr(df[col].dtype, 'name') and 'decimal' in str(df[col].dtype).lower():
                df[col] = df[col].astype(float)
        return df

    @staticmethod
    def numeric_summary(series: pd.Series) -> Dict[str, float]:
        """
        Compute summary statistics for a numeric series.

        Args:
            series: Pandas Series with numeric data

        Returns:
            Dictionary of summary statistics
        """
        if series.empty or series.isna().all():
            return {}

        return {
            'count': int(series.count()),
            'mean': round(series.mean(), 2),
            'std': round(series.std(), 2),
            'min': round(series.min(), 2),
            'q25': round(series.quantile(0.25), 2),
            'median': round(series.median(), 2),
            'q75': round(series.quantile(0.75), 2),
            'max': round(series.max(), 2),
            'skewness': round(series.skew(), 3),
            'kurtosis': round(series.kurtosis(), 3),
            'iqr': round(series.quantile(0.75) - series.quantile(0.25), 2),
            'range': round(series.max() - series.min(), 2),
            'cv': round(series.std() / series.mean() * 100, 2) if series.mean() != 0 else None
        }

    @staticmethod
    def categorical_summary(series: pd.Series) -> Dict[str, Any]:
        """
        Compute summary statistics for a categorical series.

        Args:
            series: Pandas Series with categorical data

        Returns:
            Dictionary of summary statistics
        """
        if series.empty:
            return {}

        value_counts = series.value_counts()

        return {
            'count': int(series.count()),
            'unique': int(series.nunique()),
            'top': value_counts.index[0] if len(value_counts) > 0 else None,
            'top_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'top_pct': round(value_counts.iloc[0] / len(series) * 100, 1) if len(series) > 0 else 0,
            'distribution': value_counts.head(10).to_dict()
        }

    @staticmethod
    def time_series_summary(df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """
        Compute time series statistics.

        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            value_col: Name of value column

        Returns:
            Dictionary of time series statistics
        """
        if df.empty:
            return {}

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        values = df[value_col].dropna()

        if len(values) < 2:
            return {}

        # Calculate changes
        changes = values.diff().dropna()

        return {
            'start_date': df[date_col].min().strftime('%Y-%m-%d'),
            'end_date': df[date_col].max().strftime('%Y-%m-%d'),
            'n_observations': len(values),
            'first_value': round(values.iloc[0], 2),
            'last_value': round(values.iloc[-1], 2),
            'total_change': round(values.iloc[-1] - values.iloc[0], 2),
            'pct_change': round((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100, 2) if values.iloc[0] != 0 else None,
            'avg_change': round(changes.mean(), 2),
            'volatility': round(changes.std(), 2),
            'max_increase': round(changes.max(), 2),
            'max_decrease': round(changes.min(), 2)
        }


# =============================================================================
# ELECTION STATISTICS
# =============================================================================
class ElectionStatistics(DescriptiveStatistics):
    """
    Descriptive statistics for election data.
    """

    def __init__(self, db_manager: DatabaseManager = None, data: Dict[str, pd.DataFrame] = None):
        """
        Initialize with database connection or pre-loaded data.

        Args:
            db_manager: DatabaseManager instance
            data: Pre-loaded data dictionary
        """
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self) -> None:
        """Load election data from database."""
        if self.db:
            try:
                self.data['statewide'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM ELECTION_RESULTS_STATEWIDE"
                ))
                self.data['historical'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM ELECTION_HISTORICAL"
                ))
            except Exception as e:
                logger.warning(f"Could not load from database: {e}")

    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all election statistics.

        Returns:
            Dictionary of all statistics
        """
        logger.info("Computing election statistics...")

        self.results = {
            'vote_statistics': self._vote_statistics(),
            'margin_statistics': self._margin_statistics(),
            'turnout_statistics': self._turnout_statistics(),
            'party_performance': self._party_performance(),
            'incumbent_analysis': self._incumbent_analysis(),
            'historical_trends': self._historical_trends()
        }

        return self.results

    def _vote_statistics(self) -> Dict[str, Any]:
        """Compute vote count statistics."""
        if 'statewide' not in self.data or self.data['statewide'].empty:
            return {}

        df = self.data['statewide']

        # Find vote column
        vote_col = None
        for col in ['VOTES', 'votes', 'Votes']:
            if col in df.columns:
                vote_col = col
                break

        if vote_col is None:
            return {}

        stats = {
            'overall': self.numeric_summary(df[vote_col]),
            'by_party': {},
            'by_year': {}
        }

        # By party
        party_col = 'PARTY' if 'PARTY' in df.columns else 'party'
        if party_col in df.columns:
            for party in df[party_col].unique():
                party_votes = df[df[party_col] == party][vote_col]
                stats['by_party'][party] = self.numeric_summary(party_votes)

        # By year
        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'
        if year_col in df.columns:
            for year in sorted(df[year_col].unique()):
                year_votes = df[df[year_col] == year][vote_col]
                stats['by_year'][int(year)] = {
                    'total_votes': int(year_votes.sum()),
                    'max_candidate_votes': int(year_votes.max()),
                    'candidates': int(len(year_votes))
                }

        return stats

    def _margin_statistics(self) -> Dict[str, Any]:
        """Compute victory margin statistics."""
        if 'historical' not in self.data or self.data['historical'].empty:
            return {}

        df = self.data['historical']

        margin_col = 'MARGIN_PERCENTAGE' if 'MARGIN_PERCENTAGE' in df.columns else 'margin_percentage'

        if margin_col not in df.columns:
            return {}

        margins = df[margin_col].dropna()

        stats = {
            'summary': self.numeric_summary(margins),
            'by_year': {}
        }

        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'
        if year_col in df.columns:
            for _, row in df.iterrows():
                year = int(row[year_col])
                stats['by_year'][year] = {
                    'margin_pct': round(row[margin_col], 2) if pd.notna(row[margin_col]) else None,
                    'winner': row.get('WINNER') or row.get('winner'),
                    'winner_party': row.get('WINNER_PARTY') or row.get('winner_party')
                }

        # Competitiveness analysis
        stats['competitiveness'] = {
            'closest_race': {
                'year': int(df.loc[margins.idxmin(), year_col]),
                'margin': round(margins.min(), 2)
            },
            'largest_margin': {
                'year': int(df.loc[margins.idxmax(), year_col]),
                'margin': round(margins.max(), 2)
            },
            'avg_margin': round(margins.mean(), 2),
            'races_under_10pct': int((margins < 10).sum()),
            'races_under_5pct': int((margins < 5).sum())
        }

        return stats

    def _turnout_statistics(self) -> Dict[str, Any]:
        """Compute voter turnout statistics."""
        if 'historical' not in self.data or self.data['historical'].empty:
            return {}

        df = self.data['historical']

        turnout_col = 'TURNOUT_PERCENTAGE' if 'TURNOUT_PERCENTAGE' in df.columns else 'turnout_percentage'
        total_col = 'TOTAL_VOTES' if 'TOTAL_VOTES' in df.columns else 'total_votes'

        stats = {}

        if turnout_col in df.columns:
            turnout = df[turnout_col].dropna()
            stats['turnout_pct'] = self.numeric_summary(turnout)

        if total_col in df.columns:
            totals = df[total_col].dropna()
            stats['total_votes'] = self.numeric_summary(totals)

            # Growth analysis
            if len(totals) > 1:
                totals_sorted = df.sort_values(
                    'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'
                )[total_col]
                stats['vote_growth'] = {
                    'total_growth': int(totals_sorted.iloc[-1] - totals_sorted.iloc[0]),
                    'pct_growth': round(
                        (totals_sorted.iloc[-1] - totals_sorted.iloc[0]) / totals_sorted.iloc[0] * 100, 1
                    ),
                    'avg_growth_per_cycle': int(totals_sorted.diff().mean())
                }

        return stats

    def _party_performance(self) -> Dict[str, Any]:
        """Analyze party performance over time."""
        if 'statewide' not in self.data or self.data['statewide'].empty:
            return {}

        df = self.data['statewide']

        party_col = 'PARTY' if 'PARTY' in df.columns else 'party'
        pct_col = 'VOTE_PERCENTAGE' if 'VOTE_PERCENTAGE' in df.columns else 'vote_percentage'
        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'

        if party_col not in df.columns or pct_col not in df.columns:
            return {}

        stats = {'by_party': {}}

        for party in ['R', 'D']:
            party_df = df[df[party_col] == party]

            if party_df.empty:
                continue

            pcts = party_df[pct_col].dropna()

            stats['by_party'][party] = {
                'avg_vote_share': round(pcts.mean(), 2),
                'min_vote_share': round(pcts.min(), 2),
                'max_vote_share': round(pcts.max(), 2),
                'std_vote_share': round(pcts.std(), 2),
                'wins': int(party_df[party_df.get('WINNER', party_df.get('winner', False)) == True].shape[0]),
                'elections': int(party_df[year_col].nunique())
            }

        # Two-party vote share trend
        stats['two_party_trend'] = []
        for year in sorted(df[year_col].unique()):
            year_df = df[df[year_col] == year]
            r_pct = year_df[year_df[party_col] == 'R'][pct_col].sum()
            d_pct = year_df[year_df[party_col] == 'D'][pct_col].sum()

            if r_pct + d_pct > 0:
                stats['two_party_trend'].append({
                    'year': int(year),
                    'r_two_party_pct': round(r_pct / (r_pct + d_pct) * 100, 2),
                    'd_two_party_pct': round(d_pct / (r_pct + d_pct) * 100, 2)
                })

        return stats

    def _incumbent_analysis(self) -> Dict[str, Any]:
        """Analyze incumbent performance."""
        if 'historical' not in self.data or self.data['historical'].empty:
            return {}

        df = self.data['historical']

        inc_col = 'INCUMBENT_WON' if 'INCUMBENT_WON' in df.columns else 'incumbent_won'

        if inc_col not in df.columns:
            return {}

        incumbent_won = df[inc_col].dropna()

        return {
            'total_races': int(len(incumbent_won)),
            'incumbent_wins': int(incumbent_won.sum()),
            'incumbent_losses': int((~incumbent_won).sum()),
            'incumbent_win_rate': round(incumbent_won.mean() * 100, 1)
        }

    def _historical_trends(self) -> Dict[str, Any]:
        """Analyze historical trends."""
        if 'historical' not in self.data or self.data['historical'].empty:
            return {}

        df = self.data['historical']

        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'
        margin_col = 'MARGIN_PERCENTAGE' if 'MARGIN_PERCENTAGE' in df.columns else 'margin_percentage'

        df_sorted = df.sort_values(year_col)

        trends = {
            'elections_analyzed': int(len(df)),
            'year_range': f"{int(df[year_col].min())}-{int(df[year_col].max())}"
        }

        if margin_col in df.columns:
            margins = df_sorted[margin_col].dropna()
            if len(margins) > 1:
                # Linear trend
                x = np.arange(len(margins))
                slope, intercept = np.polyfit(x, margins, 1)

                trends['margin_trend'] = {
                    'slope': round(slope, 3),
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'interpretation': f"Margins {'widening' if slope > 0 else 'narrowing'} by ~{abs(round(slope, 1))} points per cycle"
                }

        return trends


# =============================================================================
# CAMPAIGN FINANCE STATISTICS
# =============================================================================
class CampaignFinanceStatistics(DescriptiveStatistics):
    """
    Descriptive statistics for campaign finance data.
    """

    def __init__(self, db_manager: DatabaseManager = None, data: Dict[str, pd.DataFrame] = None):
        """Initialize with database connection or pre-loaded data."""
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self) -> None:
        """Load campaign finance data from database."""
        if self.db:
            try:
                self.data['summary'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM CAMPAIGN_FINANCE_SUMMARY"
                ))
                self.data['expenditures'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM CAMPAIGN_FINANCE_EXPENDITURES"
                ))
            except Exception as e:
                logger.warning(f"Could not load from database: {e}")

    def compute_all(self) -> Dict[str, Any]:
        """Compute all campaign finance statistics."""
        logger.info("Computing campaign finance statistics...")

        self.results = {
            'fundraising_statistics': self._fundraising_statistics(),
            'spending_statistics': self._spending_statistics(),
            'contributor_statistics': self._contributor_statistics(),
            'party_comparison': self._party_comparison(),
            'money_vs_results': self._money_vs_results(),
            'expenditure_categories': self._expenditure_categories()
        }

        return self.results

    def _fundraising_statistics(self) -> Dict[str, Any]:
        """Compute fundraising statistics."""
        if 'summary' not in self.data or self.data['summary'].empty:
            return {}

        df = self.data['summary']

        raised_col = 'TOTAL_RAISED' if 'TOTAL_RAISED' in df.columns else 'total_raised'

        if raised_col not in df.columns:
            return {}

        raised = df[raised_col].dropna()

        stats = {
            'overall': self.numeric_summary(raised),
            'by_year': {},
            'by_party': {}
        }

        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'
        party_col = 'PARTY' if 'PARTY' in df.columns else 'party'

        # By year
        if year_col in df.columns:
            for year in sorted(df[year_col].unique()):
                year_df = df[df[year_col] == year]
                stats['by_year'][int(year)] = {
                    'total_raised': int(year_df[raised_col].sum()),
                    'candidates': int(len(year_df)),
                    'avg_per_candidate': int(year_df[raised_col].mean())
                }

        # By party
        if party_col in df.columns:
            for party in df[party_col].unique():
                party_raised = df[df[party_col] == party][raised_col]
                stats['by_party'][party] = {
                    'total': int(party_raised.sum()),
                    'mean': int(party_raised.mean()),
                    'max': int(party_raised.max())
                }

        return stats

    def _spending_statistics(self) -> Dict[str, Any]:
        """Compute spending statistics."""
        if 'summary' not in self.data or self.data['summary'].empty:
            return {}

        df = self.data['summary']

        spent_col = 'TOTAL_SPENT' if 'TOTAL_SPENT' in df.columns else 'total_spent'
        raised_col = 'TOTAL_RAISED' if 'TOTAL_RAISED' in df.columns else 'total_raised'

        if spent_col not in df.columns:
            return {}

        spent = df[spent_col].dropna()

        stats = {
            'overall': self.numeric_summary(spent)
        }

        # Burn rate analysis
        if raised_col in df.columns:
            df_valid = df[(df[raised_col] > 0) & (df[spent_col] > 0)].copy()
            df_valid['burn_rate'] = df_valid[spent_col] / df_valid[raised_col] * 100

            stats['burn_rate'] = {
                'mean': round(df_valid['burn_rate'].mean(), 1),
                'min': round(df_valid['burn_rate'].min(), 1),
                'max': round(df_valid['burn_rate'].max(), 1)
            }

        return stats

    def _contributor_statistics(self) -> Dict[str, Any]:
        """Compute contributor statistics."""
        if 'summary' not in self.data or self.data['summary'].empty:
            return {}

        df = self.data['summary']

        contrib_col = 'NUM_CONTRIBUTORS' if 'NUM_CONTRIBUTORS' in df.columns else 'num_contributors'
        avg_col = 'AVG_CONTRIBUTION' if 'AVG_CONTRIBUTION' in df.columns else 'avg_contribution'

        stats = {}

        if contrib_col in df.columns:
            contributors = df[contrib_col].dropna()
            stats['num_contributors'] = self.numeric_summary(contributors)

        if avg_col in df.columns:
            avg_contrib = df[avg_col].dropna()
            stats['avg_contribution'] = self.numeric_summary(avg_contrib)

        # Grassroots vs big donor analysis
        if avg_col in df.columns:
            df_valid = df[df[avg_col].notna()].copy()

            stats['donor_type_analysis'] = {
                'grassroots_campaigns': int((df_valid[avg_col] < 500).sum()),
                'big_donor_campaigns': int((df_valid[avg_col] >= 1000).sum()),
                'avg_contribution_threshold': 500
            }

        return stats

    def _party_comparison(self) -> Dict[str, Any]:
        """Compare parties on fundraising metrics."""
        if 'summary' not in self.data or self.data['summary'].empty:
            return {}

        df = self.data['summary']

        party_col = 'PARTY' if 'PARTY' in df.columns else 'party'
        raised_col = 'TOTAL_RAISED' if 'TOTAL_RAISED' in df.columns else 'total_raised'
        contrib_col = 'NUM_CONTRIBUTORS' if 'NUM_CONTRIBUTORS' in df.columns else 'num_contributors'
        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'

        if party_col not in df.columns:
            return {}

        stats = {'by_cycle': []}

        for year in sorted(df[year_col].unique()):
            year_df = df[df[year_col] == year]

            r_df = year_df[year_df[party_col] == 'R']
            d_df = year_df[year_df[party_col] == 'D']

            cycle_stats = {'year': int(year)}

            if not r_df.empty and raised_col in r_df.columns:
                cycle_stats['r_raised'] = int(r_df[raised_col].sum())
            if not d_df.empty and raised_col in d_df.columns:
                cycle_stats['d_raised'] = int(d_df[raised_col].sum())

            if 'r_raised' in cycle_stats and 'd_raised' in cycle_stats:
                cycle_stats['r_advantage'] = cycle_stats['r_raised'] - cycle_stats['d_raised']
                cycle_stats['r_ratio'] = round(cycle_stats['r_raised'] / cycle_stats['d_raised'], 2) if cycle_stats['d_raised'] > 0 else None

            if not r_df.empty and contrib_col in r_df.columns:
                cycle_stats['r_contributors'] = int(r_df[contrib_col].sum())
            if not d_df.empty and contrib_col in d_df.columns:
                cycle_stats['d_contributors'] = int(d_df[contrib_col].sum())

            stats['by_cycle'].append(cycle_stats)

        return stats

    def _money_vs_results(self) -> Dict[str, Any]:
        """Analyze relationship between money and election results."""
        if 'summary' not in self.data or self.data['summary'].empty:
            return {}

        df = self.data['summary']

        raised_col = 'TOTAL_RAISED' if 'TOTAL_RAISED' in df.columns else 'total_raised'
        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'
        party_col = 'PARTY' if 'PARTY' in df.columns else 'party'

        stats = {
            'top_fundraiser_won': 0,
            'top_fundraiser_lost': 0,
            'cycles_analyzed': 0
        }

        # Winners in Texas Governor races (R won all 2010-2022)
        winners = {2010: 'R', 2014: 'R', 2018: 'R', 2022: 'R'}

        for year in df[year_col].unique():
            year_df = df[df[year_col] == year]

            if len(year_df) < 2:
                continue

            # Find top fundraiser
            top_idx = year_df[raised_col].idxmax()
            top_party = year_df.loc[top_idx, party_col]

            winner_party = winners.get(int(year))

            if winner_party:
                stats['cycles_analyzed'] += 1
                if top_party == winner_party:
                    stats['top_fundraiser_won'] += 1
                else:
                    stats['top_fundraiser_lost'] += 1

        if stats['cycles_analyzed'] > 0:
            stats['money_win_rate'] = round(
                stats['top_fundraiser_won'] / stats['cycles_analyzed'] * 100, 1
            )

        return stats

    def _expenditure_categories(self) -> Dict[str, Any]:
        """Analyze expenditure by category."""
        if 'expenditures' not in self.data or self.data['expenditures'].empty:
            return {}

        df = self.data['expenditures']

        cat_col = 'CATEGORY' if 'CATEGORY' in df.columns else 'category'
        amt_col = 'AMOUNT' if 'AMOUNT' in df.columns else 'amount'

        if cat_col not in df.columns or amt_col not in df.columns:
            return {}

        category_totals = df.groupby(cat_col)[amt_col].sum().sort_values(ascending=False)

        total = category_totals.sum()

        stats = {
            'categories': {},
            'total_expenditures': int(total)
        }

        for cat, amount in category_totals.items():
            stats['categories'][cat] = {
                'amount': int(amount),
                'percentage': round(amount / total * 100, 1)
            }

        return stats


# =============================================================================
# POLLING STATISTICS
# =============================================================================
class PollingStatistics(DescriptiveStatistics):
    """
    Descriptive statistics for polling data.
    """

    def __init__(self, db_manager: DatabaseManager = None, data: Dict[str, pd.DataFrame] = None):
        """Initialize with database connection or pre-loaded data."""
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self) -> None:
        """Load polling data from database."""
        if self.db:
            try:
                self.data['polls'] = self.convert_decimals(self.db.run_query("SELECT * FROM POLLS"))
                self.data['averages'] = self.convert_decimals(self.db.run_query("SELECT * FROM POLL_AVERAGES"))
                self.data['pollsters'] = self.convert_decimals(self.db.run_query("SELECT * FROM POLLSTERS"))
                self.data['trends'] = self.convert_decimals(self.db.run_query("SELECT * FROM POLL_TRENDS"))
            except Exception as e:
                logger.warning(f"Could not load from database: {e}")

    def compute_all(self) -> Dict[str, Any]:
        """Compute all polling statistics."""
        logger.info("Computing polling statistics...")

        self.results = {
            'poll_summary': self._poll_summary(),
            'margin_statistics': self._margin_statistics(),
            'polling_accuracy': self._polling_accuracy(),
            'pollster_analysis': self._pollster_analysis(),
            'sample_size_analysis': self._sample_size_analysis(),
            'trend_analysis': self._trend_analysis()
        }

        return self.results

    def _poll_summary(self) -> Dict[str, Any]:
        """Compute overall poll summary statistics."""
        if 'polls' not in self.data or self.data['polls'].empty:
            return {}

        df = self.data['polls']

        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'

        stats = {
            'total_polls': int(len(df)),
            'unique_pollsters': int(df['POLLSTER' if 'POLLSTER' in df.columns else 'pollster'].nunique()),
            'by_year': {}
        }

        if year_col in df.columns:
            for year in sorted(df[year_col].unique()):
                year_df = df[df[year_col] == year]
                stats['by_year'][int(year)] = {
                    'num_polls': int(len(year_df)),
                    'pollsters': int(year_df['POLLSTER' if 'POLLSTER' in df.columns else 'pollster'].nunique())
                }

        return stats

    def _margin_statistics(self) -> Dict[str, Any]:
        """Compute poll margin statistics."""
        if 'polls' not in self.data or self.data['polls'].empty:
            return {}

        df = self.data['polls']

        margin_col = 'MARGIN' if 'MARGIN' in df.columns else 'margin'
        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'

        if margin_col not in df.columns:
            return {}

        margins = df[margin_col].dropna()

        stats = {
            'overall': self.numeric_summary(margins),
            'by_year': {}
        }

        if year_col in df.columns:
            for year in sorted(df[year_col].unique()):
                year_margins = df[df[year_col] == year][margin_col].dropna()
                stats['by_year'][int(year)] = self.numeric_summary(year_margins)

        return stats

    def _polling_accuracy(self) -> Dict[str, Any]:
        """Analyze polling accuracy vs actual results."""
        if 'averages' not in self.data or self.data['averages'].empty:
            return {}

        df = self.data['averages']

        error_col = 'POLLING_ERROR' if 'POLLING_ERROR' in df.columns else 'polling_error'
        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'

        if error_col not in df.columns:
            return {}

        # Filter to final averages if available
        period_col = 'PERIOD' if 'PERIOD' in df.columns else 'period'
        if period_col in df.columns:
            final_df = df[df[period_col].str.contains('Final', case=False, na=False)]
            if final_df.empty:
                final_df = df

        errors = final_df[error_col].dropna()

        stats = {
            'overall': {
                'mean_error': round(errors.mean(), 2),
                'mean_absolute_error': round(errors.abs().mean(), 2),
                'max_error': round(errors.max(), 2),
                'min_error': round(errors.min(), 2),
                'direction': 'Underestimated R margin' if errors.mean() > 0 else 'Overestimated R margin'
            },
            'by_year': {}
        }

        if year_col in df.columns:
            for _, row in final_df.iterrows():
                year = int(row[year_col])
                stats['by_year'][year] = {
                    'polling_error': round(row[error_col], 2) if pd.notna(row[error_col]) else None,
                    'direction': 'Under R' if row[error_col] > 0 else 'Over R' if row[error_col] < 0 else 'Accurate'
                }

        return stats

    def _pollster_analysis(self) -> Dict[str, Any]:
        """Analyze pollster characteristics."""
        if 'polls' not in self.data or self.data['polls'].empty:
            return {}

        df = self.data['polls']

        pollster_col = 'POLLSTER' if 'POLLSTER' in df.columns else 'pollster'

        pollster_counts = df[pollster_col].value_counts()

        stats = {
            'most_active_pollsters': pollster_counts.head(10).to_dict(),
            'total_pollsters': int(df[pollster_col].nunique())
        }

        # Add pollster ratings if available
        if 'pollsters' in self.data and not self.data['pollsters'].empty:
            pollster_df = self.data['pollsters']
            rating_col = 'FIVETHIRTYEIGHT_RATING' if 'FIVETHIRTYEIGHT_RATING' in pollster_df.columns else 'fivethirtyeight_rating'

            if rating_col in pollster_df.columns:
                rating_counts = pollster_df[rating_col].value_counts()
                stats['rating_distribution'] = rating_counts.to_dict()

        return stats

    def _sample_size_analysis(self) -> Dict[str, Any]:
        """Analyze poll sample sizes."""
        if 'polls' not in self.data or self.data['polls'].empty:
            return {}

        df = self.data['polls']

        sample_col = 'SAMPLE_SIZE' if 'SAMPLE_SIZE' in df.columns else 'sample_size'

        if sample_col not in df.columns:
            return {}

        samples = df[sample_col].dropna()

        stats = {
            'summary': self.numeric_summary(samples),
            'size_categories': {
                'small (<500)': int((samples < 500).sum()),
                'medium (500-1000)': int(((samples >= 500) & (samples < 1000)).sum()),
                'large (1000+)': int((samples >= 1000).sum())
            }
        }

        return stats

    def _trend_analysis(self) -> Dict[str, Any]:
        """Analyze polling trends."""
        if 'trends' not in self.data or self.data['trends'].empty:
            return {}

        df = self.data['trends']

        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'
        vol_col = 'VOLATILITY' if 'VOLATILITY' in df.columns else 'volatility'
        trend_col = 'TREND_DIRECTION' if 'TREND_DIRECTION' in df.columns else 'trend_direction'

        stats = {'by_year': {}}

        for _, row in df.iterrows():
            year = int(row[year_col])
            stats['by_year'][year] = {
                'initial_margin': row.get('INITIAL_MARGIN') or row.get('initial_margin'),
                'final_margin': row.get('FINAL_MARGIN') or row.get('final_margin'),
                'volatility': round(row[vol_col], 2) if pd.notna(row.get(vol_col)) else None,
                'trend': row.get(trend_col) or row.get('trend_direction')
            }

        if vol_col in df.columns:
            stats['volatility_summary'] = self.numeric_summary(df[vol_col].dropna())

        return stats


# =============================================================================
# NEWS STATISTICS
# =============================================================================
class NewsStatistics(DescriptiveStatistics):
    """
    Descriptive statistics for news coverage data.
    """

    def __init__(self, db_manager: DatabaseManager = None, data: Dict[str, pd.DataFrame] = None):
        """Initialize with database connection or pre-loaded data."""
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self) -> None:
        """Load news data from database."""
        if self.db:
            try:
                self.data['articles'] = self.convert_decimals(self.db.run_query("SELECT * FROM NEWS_ARTICLES"))
                self.data['coverage'] = self.convert_decimals(self.db.run_query("SELECT * FROM NEWS_COVERAGE_SUMMARY"))
                self.data['by_topic'] = self.convert_decimals(self.db.run_query("SELECT * FROM NEWS_BY_TOPIC"))
            except Exception as e:
                logger.warning(f"Could not load from database: {e}")

    def compute_all(self) -> Dict[str, Any]:
        """Compute all news statistics."""
        logger.info("Computing news coverage statistics...")

        self.results = {
            'coverage_summary': self._coverage_summary(),
            'source_analysis': self._source_analysis(),
            'topic_analysis': self._topic_analysis(),
            'candidate_comparison': self._candidate_comparison(),
            'scope_analysis': self._scope_analysis()
        }

        return self.results

    def _coverage_summary(self) -> Dict[str, Any]:
        """Compute overall coverage summary."""
        if 'articles' not in self.data or self.data['articles'].empty:
            return {}

        df = self.data['articles']

        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'

        stats = {
            'total_articles': int(len(df)),
            'by_year': {}
        }

        if year_col in df.columns:
            for year in sorted(df[year_col].unique()):
                year_df = df[df[year_col] == year]
                stats['by_year'][int(year)] = {
                    'articles': int(len(year_df)),
                    'sources': int(year_df['SOURCE' if 'SOURCE' in df.columns else 'source'].nunique())
                }

        return stats

    def _source_analysis(self) -> Dict[str, Any]:
        """Analyze news sources."""
        if 'articles' not in self.data or self.data['articles'].empty:
            return {}

        df = self.data['articles']

        source_col = 'SOURCE' if 'SOURCE' in df.columns else 'source'

        source_counts = df[source_col].value_counts()

        stats = {
            'source_distribution': source_counts.to_dict(),
            'total_sources': int(df[source_col].nunique()),
            'top_source': source_counts.index[0] if len(source_counts) > 0 else None,
            'top_source_articles': int(source_counts.iloc[0]) if len(source_counts) > 0 else 0
        }

        return stats

    def _topic_analysis(self) -> Dict[str, Any]:
        """Analyze news topics."""
        if 'articles' not in self.data or self.data['articles'].empty:
            return {}

        df = self.data['articles']

        topic_col = 'TOPIC' if 'TOPIC' in df.columns else 'topic'

        if topic_col not in df.columns:
            return {}

        topic_counts = df[topic_col].value_counts()

        stats = {
            'topic_distribution': topic_counts.to_dict(),
            'total_topics': int(df[topic_col].nunique()),
            'top_topics': topic_counts.head(5).to_dict()
        }

        return stats

    def _candidate_comparison(self) -> Dict[str, Any]:
        """Compare news coverage by candidate."""
        if 'coverage' not in self.data or self.data['coverage'].empty:
            return {}

        df = self.data['coverage']

        cand_col = 'CANDIDATE' if 'CANDIDATE' in df.columns else 'candidate'
        articles_col = 'TOTAL_ARTICLES' if 'TOTAL_ARTICLES' in df.columns else 'total_articles'
        year_col = 'ELECTION_YEAR' if 'ELECTION_YEAR' in df.columns else 'election_year'

        stats = {'by_year': {}}

        for year in sorted(df[year_col].unique()):
            year_df = df[df[year_col] == year]
            year_stats = {}

            for _, row in year_df.iterrows():
                candidate = row[cand_col]
                year_stats[candidate] = {
                    'articles': int(row[articles_col]) if pd.notna(row.get(articles_col)) else 0,
                    'party': row.get('PARTY') or row.get('party')
                }

            # Calculate coverage ratio
            candidates = list(year_stats.keys())
            if len(candidates) == 2:
                a1 = year_stats[candidates[0]].get('articles', 0)
                a2 = year_stats[candidates[1]].get('articles', 0)
                if a2 > 0:
                    year_stats['coverage_ratio'] = round(a1 / a2, 2)

            stats['by_year'][int(year)] = year_stats

        return stats

    def _scope_analysis(self) -> Dict[str, Any]:
        """Analyze Texas vs National coverage."""
        if 'articles' not in self.data or self.data['articles'].empty:
            return {}

        df = self.data['articles']

        scope_col = 'SCOPE' if 'SCOPE' in df.columns else 'scope'

        if scope_col not in df.columns:
            return {}

        scope_counts = df[scope_col].value_counts()

        total = len(df)

        stats = {
            'scope_distribution': scope_counts.to_dict(),
            'texas_coverage': int(scope_counts.get('Texas', 0)),
            'national_coverage': int(scope_counts.get('National', 0)),
            'texas_pct': round(scope_counts.get('Texas', 0) / total * 100, 1) if total > 0 else 0,
            'national_pct': round(scope_counts.get('National', 0) / total * 100, 1) if total > 0 else 0
        }

        return stats


# =============================================================================
# CULTURE WAR STATISTICS
# =============================================================================
class CultureWarStatistics(DescriptiveStatistics):
    """
    Computes descriptive statistics for culture war companies data.
    """

    def __init__(self, db_manager=None, data: Dict[str, pd.DataFrame] = None):
        """
        Initialize Culture War Statistics.

        Args:
            db_manager: DatabaseManager instance for Snowflake queries
            data: Dictionary of DataFrames with culture war data
        """
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self) -> None:
        """Load culture war data from database."""
        if self.db:
            try:
                self.data['events'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM CULTURE_WAR_EVENTS"
                ))
                self.data['stock_impact'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM CULTURE_WAR_STOCK_IMPACT"
                ))
                self.data['summary'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM CULTURE_WAR_SUMMARY"
                ))
            except Exception as e:
                logger.warning(f"Could not load from database: {e}")

    def load_from_csv(self, csv_path: str = 'Culture_War_Companies_160_fullmeta.csv') -> None:
        """Load culture war data from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
            self.data['events'] = df
            logger.info(f"Loaded {len(df)} culture war events from CSV")
        except Exception as e:
            logger.warning(f"Could not load culture war CSV: {e}")

    def compute_all(self) -> Dict[str, Any]:
        """Compute all culture war statistics."""
        logger.info("Computing culture war statistics...")

        self.results = {
            'event_summary': self._event_summary(),
            'industry_analysis': self._industry_analysis(),
            'political_leaning_analysis': self._political_leaning_analysis(),
            'temporal_analysis': self._temporal_analysis(),
            'company_analysis': self._company_analysis()
        }

        return self.results

    def _event_summary(self) -> Dict[str, Any]:
        """Compute overall event summary statistics."""
        if 'events' not in self.data or self.data['events'] is None:
            return {}

        df = self.data['events']
        if df.empty:
            return {}

        # Handle column name variations
        company_col = 'COMPANY' if 'COMPANY' in df.columns else 'Company'
        industry_col = 'INDUSTRY' if 'INDUSTRY' in df.columns else 'Industry'
        ticker_col = 'TICKER' if 'TICKER' in df.columns else 'Ticker'
        year_col = 'YEAR' if 'YEAR' in df.columns else 'Year'

        stats = {
            'total_events': len(df),
            'unique_companies': df[company_col].nunique() if company_col in df.columns else 0,
            'unique_industries': df[industry_col].nunique() if industry_col in df.columns else 0,
            'year_range': {
                'min': int(df[year_col].min()) if year_col in df.columns else None,
                'max': int(df[year_col].max()) if year_col in df.columns else None
            },
            'events_per_year': df[year_col].value_counts().to_dict() if year_col in df.columns else {}
        }

        return stats

    def _industry_analysis(self) -> Dict[str, Any]:
        """Analyze culture war events by industry."""
        if 'events' not in self.data or self.data['events'] is None:
            return {}

        df = self.data['events']
        if df.empty:
            return {}

        industry_col = 'INDUSTRY' if 'INDUSTRY' in df.columns else 'Industry'

        if industry_col not in df.columns:
            return {}

        industry_counts = df[industry_col].value_counts()

        stats = {
            'industry_distribution': industry_counts.to_dict(),
            'top_industries': industry_counts.head(10).to_dict(),
            'industry_count': len(industry_counts)
        }

        return stats

    def _political_leaning_analysis(self) -> Dict[str, Any]:
        """Analyze events by political leaning."""
        if 'events' not in self.data or self.data['events'] is None:
            return {}

        df = self.data['events']
        if df.empty:
            return {}

        leaning_col = 'ESTIMATED_POLITICAL_LEANING' if 'ESTIMATED_POLITICAL_LEANING' in df.columns else 'Estimated Political Leaning'

        if leaning_col not in df.columns:
            return {}

        leaning_counts = df[leaning_col].value_counts()

        stats = {
            'leaning_distribution': leaning_counts.to_dict(),
            'liberal_events': int(leaning_counts.get('Liberal', 0)),
            'conservative_events': int(leaning_counts.get('Conservative', 0)),
            'mixed_events': int(leaning_counts.get('Mixed', 0))
        }

        # Calculate percentages
        total = len(df)
        if total > 0:
            stats['liberal_pct'] = round(stats['liberal_events'] / total * 100, 1)
            stats['conservative_pct'] = round(stats['conservative_events'] / total * 100, 1)
            stats['mixed_pct'] = round(stats['mixed_events'] / total * 100, 1)

        return stats

    def _temporal_analysis(self) -> Dict[str, Any]:
        """Analyze temporal patterns in culture war events."""
        if 'events' not in self.data or self.data['events'] is None:
            return {}

        df = self.data['events']
        if df.empty:
            return {}

        year_col = 'YEAR' if 'YEAR' in df.columns else 'Year'
        date_col = 'EVENT_DATE' if 'EVENT_DATE' in df.columns else 'Event Date'

        if year_col not in df.columns:
            return {}

        yearly_counts = df[year_col].value_counts().sort_index()

        stats = {
            'yearly_trend': yearly_counts.to_dict(),
            'peak_year': int(yearly_counts.idxmax()) if not yearly_counts.empty else None,
            'peak_year_count': int(yearly_counts.max()) if not yearly_counts.empty else 0,
            'avg_events_per_year': round(yearly_counts.mean(), 1) if not yearly_counts.empty else 0
        }

        # Growth analysis
        if len(yearly_counts) >= 2:
            years = sorted(yearly_counts.index)
            first_half = yearly_counts[yearly_counts.index <= years[len(years)//2]].mean()
            second_half = yearly_counts[yearly_counts.index > years[len(years)//2]].mean()
            stats['trend_direction'] = 'Increasing' if second_half > first_half else 'Decreasing'

        return stats

    def _company_analysis(self) -> Dict[str, Any]:
        """Analyze companies involved in culture war events."""
        if 'events' not in self.data or self.data['events'] is None:
            return {}

        df = self.data['events']
        if df.empty:
            return {}

        company_col = 'COMPANY' if 'COMPANY' in df.columns else 'Company'
        ticker_col = 'TICKER' if 'TICKER' in df.columns else 'Ticker'
        industry_col = 'INDUSTRY' if 'INDUSTRY' in df.columns else 'Industry'

        if company_col not in df.columns:
            return {}

        company_counts = df[company_col].value_counts()

        stats = {
            'companies_with_multiple_events': int((company_counts > 1).sum()),
            'top_companies': company_counts.head(10).to_dict(),
            'single_event_companies': int((company_counts == 1).sum())
        }

        # Get unique company info
        if ticker_col in df.columns:
            stats['unique_tickers'] = df[ticker_col].nunique()

        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get high-level summary for display."""
        if not self.results:
            self.compute_all()

        event_summary = self.results.get('event_summary', {})
        political = self.results.get('political_leaning_analysis', {})
        temporal = self.results.get('temporal_analysis', {})

        return {
            'total_events': event_summary.get('total_events', 'N/A'),
            'unique_companies': event_summary.get('unique_companies', 'N/A'),
            'unique_industries': event_summary.get('unique_industries', 'N/A'),
            'liberal_events': political.get('liberal_events', 'N/A'),
            'conservative_events': political.get('conservative_events', 'N/A'),
            'peak_year': temporal.get('peak_year', 'N/A'),
            'trend': temporal.get('trend_direction', 'N/A')
        }


# =============================================================================
# MARKET STATISTICS
# =============================================================================
class MarketStatistics(DescriptiveStatistics):
    """
    Computes descriptive statistics for market data (stocks, VIX, Fama-French factors).
    """

    def __init__(self, db_manager=None, data: Dict[str, pd.DataFrame] = None):
        """
        Initialize Market Statistics.

        Args:
            db_manager: DatabaseManager instance for Snowflake queries
            data: Dictionary of DataFrames with market data
        """
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self) -> None:
        """Load market data from database."""
        if self.db:
            try:
                self.data['vix'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM VIX_DAILY ORDER BY DATE"
                ))
                self.data['ff3'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM FAMA_FRENCH_FF3 ORDER BY DATE"
                ))
                self.data['stock_prices'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM STOCK_DAILY_PRICES ORDER BY DATE"
                ))
            except Exception as e:
                logger.warning(f"Could not load market data from database: {e}")

    def load_from_files(self) -> None:
        """Load market data from local files."""
        try:
            # Try to load VIX data
            vix_path = './vix_data_2000_2025.csv'
            if os.path.exists(vix_path):
                self.data['vix'] = pd.read_csv(vix_path)
                logger.info(f"Loaded VIX data from {vix_path}")
        except Exception as e:
            logger.warning(f"Could not load VIX data: {e}")

    def compute_all(self) -> Dict[str, Any]:
        """Compute all market statistics."""
        logger.info("Computing market statistics...")

        self.results = {
            'vix_analysis': self._vix_analysis(),
            'factor_analysis': self._factor_analysis(),
            'stock_analysis': self._stock_analysis()
        }

        return self.results

    def _vix_analysis(self) -> Dict[str, Any]:
        """Analyze VIX volatility data."""
        if 'vix' not in self.data or self.data['vix'] is None:
            return {}

        df = self.data['vix']
        if df.empty:
            return {}

        vix_col = 'VIX' if 'VIX' in df.columns else 'vix'

        if vix_col not in df.columns:
            return {}

        vix_series = pd.to_numeric(df[vix_col], errors='coerce').dropna()

        stats = {
            'summary': self.numeric_summary(vix_series),
            'current': float(vix_series.iloc[-1]) if len(vix_series) > 0 else None,
            'regime': 'High Volatility' if vix_series.iloc[-1] > 20 else 'Low Volatility' if len(vix_series) > 0 else None
        }

        # High volatility days (VIX > 30)
        high_vol_days = (vix_series > 30).sum()
        stats['high_volatility_days'] = int(high_vol_days)
        stats['high_volatility_pct'] = round(high_vol_days / len(vix_series) * 100, 2) if len(vix_series) > 0 else 0

        return stats

    def _factor_analysis(self) -> Dict[str, Any]:
        """Analyze Fama-French factors."""
        if 'ff3' not in self.data or self.data['ff3'] is None:
            return {}

        df = self.data['ff3']
        if df.empty:
            return {}

        stats = {}

        for col in ['MKT_RF', 'SMB', 'HML', 'mkt_rf', 'smb', 'hml']:
            if col in df.columns:
                col_upper = col.upper()
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 0:
                    stats[col_upper] = {
                        'mean': round(series.mean(), 4),
                        'std': round(series.std(), 4),
                        'sharpe': round(series.mean() / series.std() * np.sqrt(252), 2) if series.std() > 0 else 0
                    }

        return stats

    def _stock_analysis(self) -> Dict[str, Any]:
        """Analyze stock price data."""
        if 'stock_prices' not in self.data or self.data['stock_prices'] is None:
            return {}

        df = self.data['stock_prices']
        if df.empty:
            return {}

        ticker_col = 'TICKER' if 'TICKER' in df.columns else 'ticker'

        if ticker_col not in df.columns:
            return {}

        stats = {
            'unique_tickers': df[ticker_col].nunique(),
            'total_records': len(df)
        }

        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get high-level summary for display."""
        if not self.results:
            self.compute_all()

        vix = self.results.get('vix_analysis', {})

        return {
            'vix_mean': vix.get('summary', {}).get('mean', 'N/A'),
            'vix_current': vix.get('current', 'N/A'),
            'vix_regime': vix.get('regime', 'N/A'),
            'high_vol_pct': vix.get('high_volatility_pct', 'N/A')
        }


# =============================================================================
# MACROECONOMIC STATISTICS
# =============================================================================
class MacroeconomicStatistics(DescriptiveStatistics):
    """
    Computes descriptive statistics for macroeconomic data.
    """

    def __init__(self, db_manager=None, data: Dict[str, pd.DataFrame] = None):
        """
        Initialize Macroeconomic Statistics.

        Args:
            db_manager: DatabaseManager instance for Snowflake queries
            data: Dictionary of DataFrames with macroeconomic data
        """
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self) -> None:
        """Load macroeconomic data from database."""
        if self.db:
            try:
                self.data['cpi'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM MACRO_CPI ORDER BY DATE"
                ))
                self.data['gdp'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM MACRO_GDP ORDER BY DATE"
                ))
                self.data['unemployment'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM MACRO_UNEMPLOYMENT ORDER BY DATE"
                ))
                self.data['treasury_yields'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM MACRO_TREASURY_YIELDS ORDER BY DATE"
                ))
                self.data['fed_funds'] = self.convert_decimals(self.db.run_query(
                    "SELECT * FROM MACRO_POLICY_RATES ORDER BY DATE"
                ))
            except Exception as e:
                logger.warning(f"Could not load macro data from database: {e}")

    def load_from_files(self) -> None:
        """Load macroeconomic data from local files."""
        try:
            macro_path = './full_macro_data_2000_2025.csv'
            if os.path.exists(macro_path):
                self.data['macro'] = pd.read_csv(macro_path)
                logger.info(f"Loaded macro data from {macro_path}")
        except Exception as e:
            logger.warning(f"Could not load macro data: {e}")

    def compute_all(self) -> Dict[str, Any]:
        """Compute all macroeconomic statistics."""
        logger.info("Computing macroeconomic statistics...")

        self.results = {
            'inflation_analysis': self._inflation_analysis(),
            'gdp_analysis': self._gdp_analysis(),
            'employment_analysis': self._employment_analysis(),
            'rates_analysis': self._rates_analysis()
        }

        return self.results

    def _inflation_analysis(self) -> Dict[str, Any]:
        """Analyze inflation data."""
        if 'cpi' not in self.data or self.data['cpi'] is None:
            # Try macro file
            if 'macro' in self.data and self.data['macro'] is not None:
                df = self.data['macro']
                for col in ['CPI_YOY', 'cpi_yoy', 'CPIAUCSL']:
                    if col in df.columns:
                        series = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(series) > 0:
                            return {
                                'cpi_summary': self.numeric_summary(series),
                                'current': float(series.iloc[-1]),
                                'trend': 'Rising' if series.iloc[-1] > series.iloc[-12] else 'Falling' if len(series) > 12 else 'Unknown'
                            }
            return {}

        df = self.data['cpi']
        if df.empty:
            return {}

        cpi_col = 'CPI_YOY' if 'CPI_YOY' in df.columns else 'cpi_yoy'

        if cpi_col not in df.columns:
            return {}

        cpi_series = pd.to_numeric(df[cpi_col], errors='coerce').dropna()

        stats = {
            'cpi_summary': self.numeric_summary(cpi_series),
            'current': float(cpi_series.iloc[-1]) if len(cpi_series) > 0 else None,
            'trend': 'Rising' if len(cpi_series) > 12 and cpi_series.iloc[-1] > cpi_series.iloc[-12] else 'Falling'
        }

        return stats

    def _gdp_analysis(self) -> Dict[str, Any]:
        """Analyze GDP data."""
        if 'gdp' not in self.data or self.data['gdp'] is None:
            return {}

        df = self.data['gdp']
        if df.empty:
            return {}

        growth_col = 'GDP_GROWTH_YOY' if 'GDP_GROWTH_YOY' in df.columns else 'gdp_growth_yoy'

        if growth_col not in df.columns:
            return {}

        growth_series = pd.to_numeric(df[growth_col], errors='coerce').dropna()

        stats = {
            'growth_summary': self.numeric_summary(growth_series),
            'current_growth': float(growth_series.iloc[-1]) if len(growth_series) > 0 else None,
            'recession_quarters': int((growth_series < 0).sum()),
            'expansion_quarters': int((growth_series >= 0).sum())
        }

        return stats

    def _employment_analysis(self) -> Dict[str, Any]:
        """Analyze employment data."""
        if 'unemployment' not in self.data or self.data['unemployment'] is None:
            return {}

        df = self.data['unemployment']
        if df.empty:
            return {}

        unemp_col = 'UNEMPLOYMENT_RATE_U3' if 'UNEMPLOYMENT_RATE_U3' in df.columns else 'unemployment_rate_u3'

        if unemp_col not in df.columns:
            return {}

        unemp_series = pd.to_numeric(df[unemp_col], errors='coerce').dropna()

        stats = {
            'unemployment_summary': self.numeric_summary(unemp_series),
            'current_rate': float(unemp_series.iloc[-1]) if len(unemp_series) > 0 else None,
            'peak_rate': float(unemp_series.max()),
            'trough_rate': float(unemp_series.min())
        }

        return stats

    def _rates_analysis(self) -> Dict[str, Any]:
        """Analyze interest rate data."""
        if 'treasury_yields' not in self.data or self.data['treasury_yields'] is None:
            return {}

        df = self.data['treasury_yields']
        if df.empty:
            return {}

        stats = {}

        # 10-year yield analysis
        y10_col = 'YIELD_10Y' if 'YIELD_10Y' in df.columns else 'yield_10y'
        if y10_col in df.columns:
            y10_series = pd.to_numeric(df[y10_col], errors='coerce').dropna()
            if len(y10_series) > 0:
                stats['yield_10y'] = {
                    'current': float(y10_series.iloc[-1]),
                    'mean': round(y10_series.mean(), 2),
                    'min': round(y10_series.min(), 2),
                    'max': round(y10_series.max(), 2)
                }

        # Yield curve (10Y - 2Y)
        y2_col = 'YIELD_2Y' if 'YIELD_2Y' in df.columns else 'yield_2y'
        if y10_col in df.columns and y2_col in df.columns:
            spread = pd.to_numeric(df[y10_col], errors='coerce') - pd.to_numeric(df[y2_col], errors='coerce')
            spread = spread.dropna()
            if len(spread) > 0:
                stats['yield_curve'] = {
                    'current_spread': round(float(spread.iloc[-1]), 2),
                    'inverted': spread.iloc[-1] < 0,
                    'inversions': int((spread < 0).sum())
                }

        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get high-level summary for display."""
        if not self.results:
            self.compute_all()

        inflation = self.results.get('inflation_analysis', {})
        gdp = self.results.get('gdp_analysis', {})
        employment = self.results.get('employment_analysis', {})
        rates = self.results.get('rates_analysis', {})

        return {
            'current_cpi': inflation.get('current', 'N/A'),
            'current_gdp_growth': gdp.get('current_growth', 'N/A'),
            'current_unemployment': employment.get('current_rate', 'N/A'),
            'yield_10y': rates.get('yield_10y', {}).get('current', 'N/A'),
            'yield_curve_inverted': rates.get('yield_curve', {}).get('inverted', 'N/A')
        }


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
class CorrelationAnalysis:
    """
    Cross-dataset correlation analysis.
    """

    def __init__(
        self,
        election_stats: ElectionStatistics,
        finance_stats: CampaignFinanceStatistics,
        polling_stats: PollingStatistics,
        news_stats: NewsStatistics
    ):
        """Initialize with statistics objects."""
        self.election = election_stats
        self.finance = finance_stats
        self.polling = polling_stats
        self.news = news_stats
        self.results = {}

    def compute_all(self) -> Dict[str, Any]:
        """Compute all cross-dataset correlations."""
        logger.info("Computing cross-dataset correlations...")

        self.results = {
            'money_vs_votes': self._money_vs_votes(),
            'polls_vs_results': self._polls_vs_results(),
            'coverage_vs_results': self._coverage_vs_results(),
            'integrated_summary': self._integrated_summary()
        }

        return self.results

    def _money_vs_votes(self) -> Dict[str, Any]:
        """Analyze correlation between fundraising and vote share."""
        # This would require merged data from elections and finance
        return {
            'description': 'Correlation between campaign fundraising and vote percentage',
            'note': 'Requires integrated dataset with matched election-finance records'
        }

    def _polls_vs_results(self) -> Dict[str, Any]:
        """Analyze correlation between polling and actual results."""
        if not self.polling.results.get('polling_accuracy'):
            return {}

        accuracy = self.polling.results['polling_accuracy']

        return {
            'mean_absolute_error': accuracy.get('overall', {}).get('mean_absolute_error'),
            'systematic_bias': accuracy.get('overall', {}).get('direction'),
            'by_year': accuracy.get('by_year', {})
        }

    def _coverage_vs_results(self) -> Dict[str, Any]:
        """Analyze correlation between news coverage and results."""
        return {
            'description': 'Correlation between news coverage volume and election outcomes',
            'note': 'Requires integrated dataset with matched election-news records'
        }

    def _integrated_summary(self) -> Dict[str, Any]:
        """Create integrated summary across all datasets."""
        summary = {
            'election_cycles': [2010, 2014, 2018, 2022],
            'by_cycle': {}
        }

        for year in [2010, 2014, 2018, 2022]:
            cycle = {'year': year}

            # Election data
            if self.election.results.get('margin_statistics', {}).get('by_year', {}).get(year):
                cycle['election'] = self.election.results['margin_statistics']['by_year'][year]

            # Finance data
            if self.finance.results.get('fundraising_statistics', {}).get('by_year', {}).get(year):
                cycle['finance'] = self.finance.results['fundraising_statistics']['by_year'][year]

            # Polling data
            if self.polling.results.get('margin_statistics', {}).get('by_year', {}).get(year):
                cycle['polling'] = self.polling.results['margin_statistics']['by_year'][year]

            # News data
            if self.news.results.get('coverage_summary', {}).get('by_year', {}).get(year):
                cycle['news'] = self.news.results['coverage_summary']['by_year'][year]

            summary['by_cycle'][year] = cycle

        return summary


# =============================================================================
# STATISTICAL MODEL MANAGER
# =============================================================================
class StatisticalModelManager:
    """
    Manages all statistical analysis for Texas Governor race data.
    """

    def __init__(self, use_database: bool = False):
        """
        Initialize the Statistical Model Manager.

        Args:
            use_database: If True, load data from Snowflake; otherwise use ETL
        """
        self.use_database = use_database
        self.db_manager = None

        self.election_stats = None
        self.finance_stats = None
        self.polling_stats = None
        self.news_stats = None
        self.culture_war_stats = None
        self.market_stats = None
        self.macro_stats = None
        self.correlation_analysis = None

        self.all_results = {}

    def initialize(self) -> bool:
        """Initialize data sources and statistics objects."""
        if self.use_database:
            return self._initialize_from_database()
        else:
            return self._initialize_from_etl()

    def _initialize_from_database(self) -> bool:
        """Initialize using Snowflake database."""
        self.db_manager = DatabaseManager()

        if not self.db_manager.connect():
            logger.warning("Could not connect to database, falling back to ETL")
            return self._initialize_from_etl()

        self.election_stats = ElectionStatistics(db_manager=self.db_manager)
        self.finance_stats = CampaignFinanceStatistics(db_manager=self.db_manager)
        self.polling_stats = PollingStatistics(db_manager=self.db_manager)
        self.news_stats = NewsStatistics(db_manager=self.db_manager)
        self.culture_war_stats = CultureWarStatistics(db_manager=self.db_manager)
        self.market_stats = MarketStatistics(db_manager=self.db_manager)
        self.macro_stats = MacroeconomicStatistics(db_manager=self.db_manager)

        # Load data from database
        self.election_stats.load_data()
        self.finance_stats.load_data()
        self.polling_stats.load_data()
        self.news_stats.load_data()
        self.culture_war_stats.load_data()
        self.market_stats.load_data()
        self.macro_stats.load_data()

        # Also try to load from files if not in database
        if not self.culture_war_stats.data.get('events'):
            self.culture_war_stats.load_from_csv()
        if not self.market_stats.data.get('vix'):
            self.market_stats.load_from_files()
        if not self.macro_stats.data.get('cpi'):
            self.macro_stats.load_from_files()

        return True

    def _initialize_from_etl(self) -> bool:
        """Initialize using ETL pipeline data."""
        logger.info("Loading data from ETL pipeline...")

        pipeline = ETLPipeline()
        pipeline.run(extract=True, transform=True, load=False)

        transformed = pipeline.transformed_data

        self.election_stats = ElectionStatistics(data={
            'statewide': transformed.get('election_results'),
            'historical': transformed.get('election_historical')
        })

        self.finance_stats = CampaignFinanceStatistics(data={
            'summary': transformed.get('finance_summary'),
            'expenditures': transformed.get('finance_expenditures')
        })

        self.polling_stats = PollingStatistics(data={
            'polls': transformed.get('polls'),
            'averages': transformed.get('poll_averages'),
            'pollsters': transformed.get('pollsters'),
            'trends': transformed.get('poll_trends')
        })

        self.news_stats = NewsStatistics(data={
            'articles': transformed.get('news_articles'),
            'coverage': transformed.get('news_coverage_summary'),
            'by_topic': transformed.get('news_by_topic')
        })

        # Load culture war data from CSV
        self.culture_war_stats = CultureWarStatistics()
        self.culture_war_stats.load_from_csv()

        # Load market data from files
        self.market_stats = MarketStatistics()
        self.market_stats.load_from_files()

        # Load macro data from files
        self.macro_stats = MacroeconomicStatistics()
        self.macro_stats.load_from_files()

        return True

    def run_all_statistics(self) -> Dict[str, Any]:
        """Run all statistical analyses."""
        logger.info("=" * 60)
        logger.info("RUNNING DESCRIPTIVE STATISTICS")
        logger.info("=" * 60)

        self.all_results['elections'] = self.election_stats.compute_all()
        self.all_results['campaign_finance'] = self.finance_stats.compute_all()
        self.all_results['polling'] = self.polling_stats.compute_all()
        self.all_results['news'] = self.news_stats.compute_all()

        # Culture war statistics
        if self.culture_war_stats:
            self.all_results['culture_war'] = self.culture_war_stats.compute_all()

        # Market statistics
        if self.market_stats:
            self.all_results['market'] = self.market_stats.compute_all()

        # Macroeconomic statistics
        if self.macro_stats:
            self.all_results['macroeconomic'] = self.macro_stats.compute_all()

        # Cross-dataset correlations
        self.correlation_analysis = CorrelationAnalysis(
            self.election_stats,
            self.finance_stats,
            self.polling_stats,
            self.news_stats
        )
        self.all_results['correlations'] = self.correlation_analysis.compute_all()

        logger.info("=" * 60)
        logger.info("STATISTICS COMPLETE")
        logger.info("=" * 60)

        return self.all_results

    def print_summary(self) -> None:
        """Print formatted summary of all statistics."""
        print("\n" + "=" * 70)
        print("TEXAS GOVERNOR RACE - DESCRIPTIVE STATISTICS SUMMARY")
        print("=" * 70)

        # Election Statistics
        if 'elections' in self.all_results:
            print("\n" + "-" * 50)
            print("ELECTION STATISTICS")
            print("-" * 50)

            margin_stats = self.all_results['elections'].get('margin_statistics', {})
            if margin_stats.get('competitiveness'):
                comp = margin_stats['competitiveness']
                print(f"  Average Victory Margin: {comp.get('avg_margin', 'N/A')}%")
                print(f"  Closest Race: {comp.get('closest_race', {}).get('year', 'N/A')} "
                      f"({comp.get('closest_race', {}).get('margin', 'N/A')}%)")
                print(f"  Largest Margin: {comp.get('largest_margin', {}).get('year', 'N/A')} "
                      f"({comp.get('largest_margin', {}).get('margin', 'N/A')}%)")

            turnout = self.all_results['elections'].get('turnout_statistics', {})
            if turnout.get('turnout_pct'):
                print(f"  Average Turnout: {turnout['turnout_pct'].get('mean', 'N/A')}%")

        # Campaign Finance Statistics
        if 'campaign_finance' in self.all_results:
            print("\n" + "-" * 50)
            print("CAMPAIGN FINANCE STATISTICS")
            print("-" * 50)

            fundraising = self.all_results['campaign_finance'].get('fundraising_statistics', {})
            if fundraising.get('overall'):
                overall = fundraising['overall']
                print(f"  Average Raised per Candidate: ${overall.get('mean', 0):,.0f}")
                print(f"  Maximum Raised: ${overall.get('max', 0):,.0f}")

            money_results = self.all_results['campaign_finance'].get('money_vs_results', {})
            if money_results.get('money_win_rate'):
                print(f"  Top Fundraiser Win Rate: {money_results['money_win_rate']}%")

        # Polling Statistics
        if 'polling' in self.all_results:
            print("\n" + "-" * 50)
            print("POLLING STATISTICS")
            print("-" * 50)

            poll_summary = self.all_results['polling'].get('poll_summary', {})
            print(f"  Total Polls Analyzed: {poll_summary.get('total_polls', 'N/A')}")
            print(f"  Unique Pollsters: {poll_summary.get('unique_pollsters', 'N/A')}")

            accuracy = self.all_results['polling'].get('polling_accuracy', {})
            if accuracy.get('overall'):
                overall = accuracy['overall']
                print(f"  Mean Absolute Error: {overall.get('mean_absolute_error', 'N/A')} points")
                print(f"  Systematic Bias: {overall.get('direction', 'N/A')}")

        # News Statistics
        if 'news' in self.all_results:
            print("\n" + "-" * 50)
            print("NEWS COVERAGE STATISTICS")
            print("-" * 50)

            coverage = self.all_results['news'].get('coverage_summary', {})
            print(f"  Total Articles: {coverage.get('total_articles', 'N/A')}")

            scope = self.all_results['news'].get('scope_analysis', {})
            if scope:
                print(f"  Texas Coverage: {scope.get('texas_pct', 'N/A')}%")
                print(f"  National Coverage: {scope.get('national_pct', 'N/A')}%")

        # Culture War Statistics
        if 'culture_war' in self.all_results:
            print("\n" + "-" * 50)
            print("CULTURE WAR STATISTICS")
            print("-" * 50)

            event_summary = self.all_results['culture_war'].get('event_summary', {})
            print(f"  Total Events: {event_summary.get('total_events', 'N/A')}")
            print(f"  Unique Companies: {event_summary.get('unique_companies', 'N/A')}")
            print(f"  Unique Industries: {event_summary.get('unique_industries', 'N/A')}")

            political = self.all_results['culture_war'].get('political_leaning_analysis', {})
            if political:
                print(f"  Liberal Events: {political.get('liberal_events', 'N/A')} ({political.get('liberal_pct', 'N/A')}%)")
                print(f"  Conservative Events: {political.get('conservative_events', 'N/A')} ({political.get('conservative_pct', 'N/A')}%)")
                print(f"  Mixed Events: {political.get('mixed_events', 'N/A')} ({political.get('mixed_pct', 'N/A')}%)")

            temporal = self.all_results['culture_war'].get('temporal_analysis', {})
            if temporal:
                print(f"  Peak Year: {temporal.get('peak_year', 'N/A')} ({temporal.get('peak_year_count', 'N/A')} events)")
                print(f"  Trend: {temporal.get('trend_direction', 'N/A')}")

        # Market Statistics
        if 'market' in self.all_results:
            print("\n" + "-" * 50)
            print("MARKET STATISTICS")
            print("-" * 50)

            vix = self.all_results['market'].get('vix_analysis', {})
            if vix:
                summary = vix.get('summary', {})
                print(f"  VIX Mean: {summary.get('mean', 'N/A')}")
                print(f"  VIX Current: {vix.get('current', 'N/A')}")
                print(f"  Market Regime: {vix.get('regime', 'N/A')}")
                print(f"  High Volatility Days: {vix.get('high_volatility_pct', 'N/A')}%")

        # Macroeconomic Statistics
        if 'macroeconomic' in self.all_results:
            print("\n" + "-" * 50)
            print("MACROECONOMIC STATISTICS")
            print("-" * 50)

            inflation = self.all_results['macroeconomic'].get('inflation_analysis', {})
            if inflation:
                print(f"  Current CPI (YoY): {inflation.get('current', 'N/A')}%")
                print(f"  Inflation Trend: {inflation.get('trend', 'N/A')}")

            gdp = self.all_results['macroeconomic'].get('gdp_analysis', {})
            if gdp:
                print(f"  Current GDP Growth: {gdp.get('current_growth', 'N/A')}%")

            employment = self.all_results['macroeconomic'].get('employment_analysis', {})
            if employment:
                print(f"  Current Unemployment: {employment.get('current_rate', 'N/A')}%")

            rates = self.all_results['macroeconomic'].get('rates_analysis', {})
            if rates:
                y10 = rates.get('yield_10y', {})
                print(f"  10-Year Treasury Yield: {y10.get('current', 'N/A')}%")
                curve = rates.get('yield_curve', {})
                if curve:
                    print(f"  Yield Curve Inverted: {curve.get('inverted', 'N/A')}")

        print("\n" + "=" * 70)

    def export_results(self, output_dir: str = OUTPUT_DIR) -> Dict[str, str]:
        """
        Export all results to files.

        Args:
            output_dir: Output directory

        Returns:
            Dictionary of output file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = {}

        # Export JSON
        json_file = output_path / 'descriptive_statistics.json'
        with open(json_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        output_files['json'] = str(json_file)
        logger.info(f"Exported JSON: {json_file}")

        # Export summary report
        report_file = output_path / 'statistics_report.txt'
        with open(report_file, 'w') as f:
            f.write("TEXAS GOVERNOR RACE - DESCRIPTIVE STATISTICS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            for category, stats in self.all_results.items():
                f.write(f"\n{category.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(json.dumps(stats, indent=2, default=str))
                f.write("\n")

        output_files['report'] = str(report_file)
        logger.info(f"Exported report: {report_file}")

        return output_files

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.disconnect()


# =============================================================================
# PREDICTIVE MODELING
# =============================================================================
class PredictiveModel:
    """
    Predictive models for Texas Governor race outcomes.
    Includes OLS regression and Logistic regression with FinBERT sentiment.
    """

    def __init__(self, manager: StatisticalModelManager):
        """
        Initialize predictive model with data from StatisticalModelManager.

        Args:
            manager: Initialized StatisticalModelManager with computed statistics
        """
        self.manager = manager
        self.model_data = None
        self.ols_results = None
        self.logistic_results = None
        self.sentiment_cache = {}

    def prepare_model_data(self) -> pd.DataFrame:
        """
        Prepare consolidated dataset for modeling.
        Creates one row per election year with all features.

        Returns:
            DataFrame with features for each election year
        """
        logger.info("Preparing model data...")

        election_years = [2010, 2014, 2018, 2022]
        data_rows = []

        for year in election_years:
            row = {'election_year': year}

            # Election results (dependent variable: R won = 1)
            elections = self.manager.all_results.get('elections', {})
            margin_by_year = elections.get('margin_statistics', {}).get('by_year', {})
            year_data = margin_by_year.get(year, {})

            row['winner_r'] = 1 if year_data.get('winner_party') == 'R' else 0
            row['margin_pct'] = year_data.get('margin_pct', 0)

            # Campaign finance
            finance = self.manager.all_results.get('campaign_finance', {})
            party_comparison = finance.get('party_comparison', {}).get('by_cycle', [])

            for cycle in party_comparison:
                if cycle.get('year') == year:
                    row['r_raised'] = cycle.get('r_raised', 0)
                    row['d_raised'] = cycle.get('d_raised', 0)
                    row['r_fundraising_advantage'] = cycle.get('r_advantage', 0)
                    row['r_fundraising_ratio'] = cycle.get('r_ratio', 1)
                    break
            else:
                row['r_raised'] = 0
                row['d_raised'] = 0
                row['r_fundraising_advantage'] = 0
                row['r_fundraising_ratio'] = 1

            # Polling data
            polling = self.manager.all_results.get('polling', {})
            poll_margin_by_year = polling.get('margin_statistics', {}).get('by_year', {})
            poll_data = poll_margin_by_year.get(year, {})

            row['poll_margin_mean'] = poll_data.get('mean', 0)
            row['poll_margin_std'] = poll_data.get('std', 0)

            poll_accuracy = polling.get('polling_accuracy', {}).get('by_year', {})
            accuracy_data = poll_accuracy.get(str(year), {})
            row['historical_polling_error'] = accuracy_data.get('polling_error', 0)

            # News coverage
            news = self.manager.all_results.get('news', {})
            news_by_year = news.get('coverage_summary', {}).get('by_year', {})
            news_data = news_by_year.get(year, {})

            row['news_articles'] = news_data.get('articles', 0)
            row['news_sources'] = news_data.get('sources', 0)

            # Culture war events (count events around election year)
            culture_war = self.manager.all_results.get('culture_war', {})
            yearly_events = culture_war.get('temporal_analysis', {}).get('yearly_trend', {})
            row['culture_war_events'] = yearly_events.get(year, 0) + yearly_events.get(year - 1, 0)

            # Market data - VIX around election time
            market = self.manager.all_results.get('market', {})
            vix_summary = market.get('vix_analysis', {}).get('summary', {})
            row['vix_mean'] = vix_summary.get('mean', 20)
            row['vix_std'] = vix_summary.get('std', 5)

            # Macroeconomic data
            macro = self.manager.all_results.get('macroeconomic', {})
            inflation = macro.get('inflation_analysis', {})
            row['inflation_current'] = inflation.get('current', 2.5) if inflation.get('current') != 'N/A' else 2.5

            gdp = macro.get('gdp_analysis', {})
            row['gdp_growth'] = gdp.get('current_growth', 2.0) if gdp.get('current_growth') else 2.0

            employment = macro.get('employment_analysis', {})
            row['unemployment_rate'] = employment.get('current_rate', 5.0) if employment.get('current_rate') else 5.0

            data_rows.append(row)

        self.model_data = pd.DataFrame(data_rows)
        logger.info(f"Prepared model data with {len(self.model_data)} observations")

        return self.model_data

    def add_sentiment_analysis(self) -> pd.DataFrame:
        """
        Add FinBERT sentiment analysis for news coverage.

        Returns:
            Updated DataFrame with sentiment features
        """
        logger.info("Adding FinBERT sentiment analysis...")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            # Load FinBERT model
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Load news articles
            news_path = './data/news/texas_governor_news.csv'
            if os.path.exists(news_path):
                news_df = pd.read_csv(news_path)
            else:
                logger.warning("News CSV not found, using placeholder sentiment")
                self.model_data['news_sentiment_positive'] = 0.33
                self.model_data['news_sentiment_negative'] = 0.33
                self.model_data['news_sentiment_neutral'] = 0.34
                return self.model_data

            # Get sentiment for each election year
            for idx, row in self.model_data.iterrows():
                year = row['election_year']

                # Filter news for election year
                if 'election_year' in news_df.columns:
                    year_news = news_df[news_df['election_year'] == year]
                else:
                    year_news = news_df

                if len(year_news) == 0:
                    self.model_data.loc[idx, 'news_sentiment_positive'] = 0.33
                    self.model_data.loc[idx, 'news_sentiment_negative'] = 0.33
                    self.model_data.loc[idx, 'news_sentiment_neutral'] = 0.34
                    continue

                # Sample headlines/titles for sentiment
                text_col = 'title' if 'title' in year_news.columns else 'headline'
                if text_col not in year_news.columns:
                    text_col = year_news.columns[0]

                texts = year_news[text_col].dropna().head(50).tolist()

                if not texts:
                    self.model_data.loc[idx, 'news_sentiment_positive'] = 0.33
                    self.model_data.loc[idx, 'news_sentiment_negative'] = 0.33
                    self.model_data.loc[idx, 'news_sentiment_neutral'] = 0.34
                    continue

                # Get sentiment scores
                sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
                for text in texts:
                    try:
                        inputs = tokenizer(str(text)[:512], return_tensors="pt", truncation=True, padding=True)
                        with torch.no_grad():
                            outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        sentiments['positive'] += probs[0][0].item()
                        sentiments['negative'] += probs[0][1].item()
                        sentiments['neutral'] += probs[0][2].item()
                    except:
                        continue

                total = len(texts)
                if total > 0:
                    self.model_data.loc[idx, 'news_sentiment_positive'] = sentiments['positive'] / total
                    self.model_data.loc[idx, 'news_sentiment_negative'] = sentiments['negative'] / total
                    self.model_data.loc[idx, 'news_sentiment_neutral'] = sentiments['neutral'] / total
                else:
                    self.model_data.loc[idx, 'news_sentiment_positive'] = 0.33
                    self.model_data.loc[idx, 'news_sentiment_negative'] = 0.33
                    self.model_data.loc[idx, 'news_sentiment_neutral'] = 0.34

            logger.info("Sentiment analysis complete")

        except ImportError as e:
            logger.warning(f"FinBERT not available: {e}. Using placeholder sentiment.")
            self.model_data['news_sentiment_positive'] = 0.33
            self.model_data['news_sentiment_negative'] = 0.33
            self.model_data['news_sentiment_neutral'] = 0.34
        except Exception as e:
            logger.warning(f"Sentiment analysis error: {e}. Using placeholder sentiment.")
            self.model_data['news_sentiment_positive'] = 0.33
            self.model_data['news_sentiment_negative'] = 0.33
            self.model_data['news_sentiment_neutral'] = 0.34

        return self.model_data

    def run_ols_regression(self) -> Dict[str, Any]:
        """
        Run OLS regression to predict election winner.

        Returns:
            Dictionary with OLS results and statistics
        """
        logger.info("Running OLS regression...")

        try:
            import statsmodels.api as sm

            if self.model_data is None:
                self.prepare_model_data()

            # Define features (exclude dependent variable and identifiers)
            exclude_cols = ['election_year', 'winner_r']
            feature_cols = [c for c in self.model_data.columns if c not in exclude_cols]

            X = self.model_data[feature_cols].fillna(0)
            y = self.model_data['winner_r']

            # Add constant for OLS
            X_const = sm.add_constant(X)

            # Fit OLS model
            model = sm.OLS(y, X_const)
            results = model.fit()

            self.ols_results = {
                'model_type': 'OLS Regression',
                'r_squared': round(results.rsquared, 4),
                'adj_r_squared': round(results.rsquared_adj, 4),
                'f_statistic': round(results.fvalue, 4) if not np.isnan(results.fvalue) else None,
                'f_pvalue': round(results.f_pvalue, 4) if not np.isnan(results.f_pvalue) else None,
                'aic': round(results.aic, 2),
                'bic': round(results.bic, 2),
                'n_observations': int(results.nobs),
                'coefficients': {},
                'predictions': results.fittedvalues.tolist(),
                'residuals': results.resid.tolist()
            }

            # Extract coefficients
            for i, col in enumerate(X_const.columns):
                self.ols_results['coefficients'][col] = {
                    'coefficient': round(float(results.params.iloc[i]), 4),
                    'std_error': round(float(results.bse.iloc[i]), 4) if not np.isnan(results.bse.iloc[i]) else None,
                    't_statistic': round(float(results.tvalues.iloc[i]), 4) if not np.isnan(results.tvalues.iloc[i]) else None,
                    'p_value': round(float(results.pvalues.iloc[i]), 4) if not np.isnan(results.pvalues.iloc[i]) else None
                }

            # Calculate accuracy (predicted > 0.5 = R wins)
            predictions = (results.fittedvalues > 0.5).astype(int)
            accuracy = (predictions == y).mean()
            self.ols_results['accuracy'] = round(accuracy * 100, 2)

            logger.info(f"OLS R-squared: {self.ols_results['r_squared']}, Accuracy: {self.ols_results['accuracy']}%")

            return self.ols_results

        except ImportError:
            logger.error("statsmodels not installed")
            return {'error': 'statsmodels not installed'}
        except Exception as e:
            logger.error(f"OLS regression error: {e}")
            return {'error': str(e)}

    def run_logistic_regression(self, train_years: list = [2010, 2014], test_years: list = [2018, 2022]) -> Dict[str, Any]:
        """
        Run logistic regression with train/test split.
        Falls back to margin prediction if all outcomes are the same class.

        Args:
            train_years: Years to use for training
            test_years: Years to use for testing

        Returns:
            Dictionary with regression results
        """
        logger.info(f"Running logistic regression (train: {train_years}, test: {test_years})...")

        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error

            if self.model_data is None:
                self.prepare_model_data()

            # Add sentiment if not present
            if 'news_sentiment_positive' not in self.model_data.columns:
                self.add_sentiment_analysis()

            # Define features
            exclude_cols = ['election_year', 'winner_r', 'margin_pct']
            feature_cols = [c for c in self.model_data.columns if c not in exclude_cols]

            # Check if we have class variation
            unique_classes = self.model_data['winner_r'].nunique()

            if unique_classes < 2:
                logger.warning("Only one class in target variable. Using Ridge regression on margin instead.")
                return self._run_margin_regression(train_years, test_years, feature_cols)

            # Split data
            train_mask = self.model_data['election_year'].isin(train_years)
            test_mask = self.model_data['election_year'].isin(test_years)

            X_train = self.model_data.loc[train_mask, feature_cols].fillna(0)
            y_train = self.model_data.loc[train_mask, 'winner_r']
            X_test = self.model_data.loc[test_mask, feature_cols].fillna(0)
            y_test = self.model_data.loc[test_mask, 'winner_r']

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit logistic regression
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)

            # Predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            train_proba = model.predict_proba(X_train_scaled)[:, 1]
            test_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Results
            self.logistic_results = {
                'model_type': 'Logistic Regression',
                'train_years': train_years,
                'test_years': test_years,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'feature_importance': {},
                'training_metrics': {
                    'accuracy': round(accuracy_score(y_train, train_pred) * 100, 2)
                },
                'testing_metrics': {
                    'accuracy': round(accuracy_score(y_test, test_pred) * 100, 2)
                },
                'predictions': {
                    'train': {
                        'years': train_years,
                        'actual': y_train.tolist(),
                        'predicted': train_pred.tolist(),
                        'probability': train_proba.tolist()
                    },
                    'test': {
                        'years': test_years,
                        'actual': y_test.tolist(),
                        'predicted': test_pred.tolist(),
                        'probability': test_proba.tolist()
                    }
                }
            }

            # Feature importance (coefficients)
            for i, col in enumerate(feature_cols):
                self.logistic_results['feature_importance'][col] = round(model.coef_[0][i], 4)

            # Backfill predictions
            X_all = self.model_data[feature_cols].fillna(0)
            X_all_scaled = scaler.transform(X_all)
            all_pred = model.predict(X_all_scaled)
            all_proba = model.predict_proba(X_all_scaled)[:, 1]

            self.logistic_results['backfill'] = {
                'all_years': self.model_data['election_year'].tolist(),
                'actual': self.model_data['winner_r'].tolist(),
                'predicted': all_pred.tolist(),
                'probability': all_proba.tolist(),
                'accuracy': round(accuracy_score(self.model_data['winner_r'], all_pred) * 100, 2)
            }

            logger.info(f"Logistic regression - Train accuracy: {self.logistic_results['training_metrics']['accuracy']}%, "
                       f"Test accuracy: {self.logistic_results['testing_metrics']['accuracy']}%")

            return self.logistic_results

        except ImportError as e:
            logger.error(f"sklearn not installed: {e}")
            return {'error': 'sklearn not installed'}
        except Exception as e:
            logger.error(f"Logistic regression error: {e}")
            # Try margin regression as fallback
            try:
                return self._run_margin_regression(train_years, test_years, feature_cols)
            except:
                return {'error': str(e)}

    def _run_margin_regression(self, train_years: list, test_years: list, feature_cols: list) -> Dict[str, Any]:
        """
        Run Ridge regression to predict victory margin.
        Used when classification isn't possible (single class).
        """
        logger.info("Running margin prediction regression...")

        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # Split data
        train_mask = self.model_data['election_year'].isin(train_years)
        test_mask = self.model_data['election_year'].isin(test_years)

        X_train = self.model_data.loc[train_mask, feature_cols].fillna(0)
        y_train = self.model_data.loc[train_mask, 'margin_pct']
        X_test = self.model_data.loc[test_mask, feature_cols].fillna(0)
        y_test = self.model_data.loc[test_mask, 'margin_pct']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit Ridge regression
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        # Convert margin to win probability (margin > 0 = R wins)
        train_winner_pred = (train_pred > 0).astype(int)
        test_winner_pred = (test_pred > 0).astype(int)

        # All predictions
        X_all = self.model_data[feature_cols].fillna(0)
        X_all_scaled = scaler.transform(X_all)
        all_margin_pred = model.predict(X_all_scaled)
        all_winner_pred = (all_margin_pred > 0).astype(int)

        # Calculate win probability as sigmoid of margin
        def margin_to_prob(margin):
            return 1 / (1 + np.exp(-margin / 5))

        self.logistic_results = {
            'model_type': 'Ridge Regression (Margin Prediction)',
            'note': 'All elections won by same party - predicting margin instead of winner',
            'train_years': train_years,
            'test_years': test_years,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'feature_importance': {},
            'training_metrics': {
                'r2_score': round(r2_score(y_train, train_pred), 4),
                'mae': round(mean_absolute_error(y_train, train_pred), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_train, train_pred)), 2),
                'accuracy': round((train_winner_pred == self.model_data.loc[train_mask, 'winner_r']).mean() * 100, 2)
            },
            'testing_metrics': {
                'r2_score': round(r2_score(y_test, test_pred), 4),
                'mae': round(mean_absolute_error(y_test, test_pred), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_test, test_pred)), 2),
                'accuracy': round((test_winner_pred == self.model_data.loc[test_mask, 'winner_r']).mean() * 100, 2)
            },
            'predictions': {
                'train': {
                    'years': train_years,
                    'actual_margin': y_train.tolist(),
                    'predicted_margin': train_pred.tolist(),
                    'actual': self.model_data.loc[train_mask, 'winner_r'].tolist(),
                    'predicted': train_winner_pred.tolist(),
                    'probability': [margin_to_prob(m) for m in train_pred]
                },
                'test': {
                    'years': test_years,
                    'actual_margin': y_test.tolist(),
                    'predicted_margin': test_pred.tolist(),
                    'actual': self.model_data.loc[test_mask, 'winner_r'].tolist(),
                    'predicted': test_winner_pred.tolist(),
                    'probability': [margin_to_prob(m) for m in test_pred]
                }
            },
            'backfill': {
                'all_years': self.model_data['election_year'].tolist(),
                'actual': self.model_data['winner_r'].tolist(),
                'actual_margin': self.model_data['margin_pct'].tolist(),
                'predicted': all_winner_pred.tolist(),
                'predicted_margin': all_margin_pred.tolist(),
                'probability': [margin_to_prob(m) for m in all_margin_pred],
                'accuracy': round((all_winner_pred == self.model_data['winner_r']).mean() * 100, 2)
            }
        }

        # Feature importance
        for i, col in enumerate(feature_cols):
            self.logistic_results['feature_importance'][col] = round(model.coef_[i], 4)

        logger.info(f"Margin regression - Train R2: {self.logistic_results['training_metrics']['r2_score']}, "
                   f"Test MAE: {self.logistic_results['testing_metrics']['mae']}%")

        return self.logistic_results

    def run_all_models(self) -> Dict[str, Any]:
        """
        Run all predictive models.

        Returns:
            Dictionary with all model results
        """
        logger.info("=" * 60)
        logger.info("RUNNING PREDICTIVE MODELS")
        logger.info("=" * 60)

        self.prepare_model_data()
        self.add_sentiment_analysis()

        results = {
            'ols_regression': self.run_ols_regression(),
            'logistic_regression': self.run_logistic_regression(),
            'model_data': self.model_data.to_dict('records') if self.model_data is not None else None
        }

        logger.info("=" * 60)
        logger.info("PREDICTIVE MODELS COMPLETE")
        logger.info("=" * 60)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of model results for display."""
        summary = {}

        if self.ols_results:
            summary['ols'] = {
                'r_squared': self.ols_results.get('r_squared'),
                'accuracy': self.ols_results.get('accuracy'),
                'n_observations': self.ols_results.get('n_observations')
            }

        if self.logistic_results:
            summary['logistic'] = {
                'train_accuracy': self.logistic_results.get('training_metrics', {}).get('accuracy'),
                'test_accuracy': self.logistic_results.get('testing_metrics', {}).get('accuracy'),
                'backfill_accuracy': self.logistic_results.get('backfill', {}).get('accuracy')
            }

        return summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point for statistical analysis."""
    parser = argparse.ArgumentParser(
        description='Descriptive Statistics for Texas Governor Race Data'
    )

    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Run all summary statistics'
    )
    parser.add_argument(
        '--elections', '-e',
        action='store_true',
        help='Run election statistics only'
    )
    parser.add_argument(
        '--finance', '-f',
        action='store_true',
        help='Run campaign finance statistics only'
    )
    parser.add_argument(
        '--polling', '-p',
        action='store_true',
        help='Run polling statistics only'
    )
    parser.add_argument(
        '--news', '-n',
        action='store_true',
        help='Run news coverage statistics only'
    )
    parser.add_argument(
        '--culture-war', '-w',
        action='store_true',
        help='Run culture war statistics only'
    )
    parser.add_argument(
        '--market', '-m',
        action='store_true',
        help='Run market statistics (VIX, Fama-French) only'
    )
    parser.add_argument(
        '--macro',
        action='store_true',
        help='Run macroeconomic statistics only'
    )
    parser.add_argument(
        '--correlations', '-c',
        action='store_true',
        help='Run cross-dataset correlation analysis'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export results to files'
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--use-database',
        action='store_true',
        help='Use Snowflake database instead of ETL pipeline'
    )
    parser.add_argument(
        '--print-json',
        action='store_true',
        help='Print results as JSON'
    )

    args = parser.parse_args()

    # Default to summary if no specific option selected
    if not any([args.summary, args.elections, args.finance, args.polling, args.news, args.culture_war, args.market, args.macro, args.correlations]):
        args.summary = True

    # Initialize manager
    manager = StatisticalModelManager(use_database=args.use_database)

    if not manager.initialize():
        logger.error("Failed to initialize data sources")
        sys.exit(1)

    try:
        results = {}

        if args.summary:
            results = manager.run_all_statistics()
            manager.print_summary()

        else:
            if args.elections:
                results['elections'] = manager.election_stats.compute_all()

            if args.finance:
                results['campaign_finance'] = manager.finance_stats.compute_all()

            if args.polling:
                results['polling'] = manager.polling_stats.compute_all()

            if args.news:
                results['news'] = manager.news_stats.compute_all()

            if args.culture_war:
                results['culture_war'] = manager.culture_war_stats.compute_all()

            if args.market:
                results['market'] = manager.market_stats.compute_all()

            if args.macro:
                results['macroeconomic'] = manager.macro_stats.compute_all()

            if args.correlations:
                manager.run_all_statistics()  # Need all stats for correlations
                results['correlations'] = manager.all_results.get('correlations', {})

        if args.print_json:
            print(json.dumps(results, indent=2, default=str))

        if args.export:
            manager.all_results = results
            output_files = manager.export_results(args.output_dir)
            print(f"\nResults exported to: {args.output_dir}")

    finally:
        manager.cleanup()


if __name__ == "__main__":
    main()
