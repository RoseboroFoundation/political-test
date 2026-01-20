"""
Snowflake Database Module V2 - Multi-Race Election Framework

This module provides a generalized database schema supporting multiple race types,
states, and election cycles.

REFACTORING NOTES:
- [MODIFY] Schema name: TEXAS_GOVERNOR → ELECTION_PREDICTIONS
- [MODIFY] All tables now include RACE_TYPE, STATE, DISTRICT columns
- [ADD] New dimension tables for race types, states
- [ADD] MODEL_PREDICTIONS table for storing predictions with confidence
- [ADD] FEATURE_AVAILABILITY table for tracking data sparsity
- [KEEP] Core table structures for elections, finance, polling, news, macro

Key Schema Changes:
1. ELECTION_RESULTS_STATEWIDE → ELECTION_RESULTS (multi-race)
2. Add RACE_TYPE dimension table
3. Add MODEL_OUTPUTS table for prediction storage
4. Add TRAINING_DATA table for ML feature matrices
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

import pandas as pd

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - UPDATED FOR MULTI-RACE
# =============================================================================

# [MODIFY] Schema name changed from TEXAS_GOVERNOR
DEFAULT_SCHEMA = 'ELECTION_PREDICTIONS'
DEFAULT_WAREHOUSE = 'COMPUTE_WH'


# =============================================================================
# UPDATED TABLE DEFINITIONS - MULTI-RACE SUPPORT
# =============================================================================

SNOWFLAKE_TABLES_V2 = {
    # =========================================================================
    # DIMENSION TABLES (NEW)
    # =========================================================================

    'DIM_RACE_TYPE': {
        'source_key': None,
        'columns': {
            'RACE_TYPE_ID': 'INTEGER',
            'RACE_TYPE': 'VARCHAR(50)',       # governor, us_senate, us_house, etc.
            'RACE_LEVEL': 'VARCHAR(20)',       # federal, state, local
            'CYCLE_LENGTH': 'INTEGER',         # Years between elections
            'DESCRIPTION': 'VARCHAR(200)'
        },
        'primary_key': ['RACE_TYPE_ID'],
        'description': 'Dimension table for race types'
    },

    'DIM_STATE': {
        'source_key': None,
        'columns': {
            'STATE_CODE': 'VARCHAR(2)',
            'STATE_NAME': 'VARCHAR(50)',
            'REGION': 'VARCHAR(20)',           # Northeast, South, Midwest, West
            'PARTISAN_LEAN': 'FLOAT',          # Cook PVI-style score
            'ELECTORAL_VOTES': 'INTEGER',
            'POPULATION': 'INTEGER',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['STATE_CODE'],
        'description': 'State dimension with partisan lean'
    },

    # =========================================================================
    # ELECTION RESULTS (MODIFIED - MULTI-RACE)
    # =========================================================================

    # [MODIFY] Renamed from ELECTION_RESULTS_STATEWIDE, added RACE_TYPE
    'ELECTION_RESULTS': {
        'source_key': 'election_results',
        'columns': {
            'RESULT_ID': 'INTEGER AUTOINCREMENT',
            'ELECTION_YEAR': 'INTEGER',
            'ELECTION_DATE': 'DATE',
            'RACE_TYPE': 'VARCHAR(50)',        # [ADD] governor, us_senate, us_house
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',              # [ADD] For House races
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'VOTES': 'INTEGER',
            'VOTE_PERCENTAGE': 'FLOAT',
            'TOTAL_VOTES': 'INTEGER',
            'TURNOUT_PERCENTAGE': 'FLOAT',
            'WINNER': 'BOOLEAN',
            'INCUMBENT': 'BOOLEAN',
            'MARGIN_PCT': 'FLOAT',             # [ADD] Winner's margin
            'DATA_SOURCE': 'VARCHAR(100)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['RESULT_ID'],
        'description': 'Election results for all race types'
    },

    # [MODIFY] Historical summary with RACE_TYPE
    'ELECTION_HISTORICAL': {
        'source_key': 'election_historical',
        'columns': {
            'ELECTION_ID': 'INTEGER AUTOINCREMENT',
            'ELECTION_YEAR': 'INTEGER',
            'ELECTION_DATE': 'DATE',
            'RACE_TYPE': 'VARCHAR(50)',        # [ADD]
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',              # [ADD]
            'WINNER': 'VARCHAR(100)',
            'WINNER_PARTY': 'VARCHAR(10)',
            'WINNER_VOTES': 'INTEGER',
            'WINNER_PERCENTAGE': 'FLOAT',
            'RUNNER_UP': 'VARCHAR(100)',
            'RUNNER_UP_PARTY': 'VARCHAR(10)',
            'RUNNER_UP_VOTES': 'INTEGER',
            'RUNNER_UP_PERCENTAGE': 'FLOAT',
            'MARGIN_PERCENTAGE': 'FLOAT',
            'MARGIN_VOTES': 'INTEGER',
            'TOTAL_VOTES': 'INTEGER',
            'TURNOUT_PERCENTAGE': 'FLOAT',
            'INCUMBENT_WON': 'BOOLEAN',
            'PARTY_FLIP': 'BOOLEAN',
            'OPEN_SEAT': 'BOOLEAN',            # [ADD]
            'ELECTION_CONTEXT': 'VARCHAR(20)', # [ADD] presidential/midterm
            'PREVIOUS_MARGIN': 'FLOAT',        # [ADD]
            'MARGIN_CHANGE': 'FLOAT',
            'TURNOUT_CHANGE': 'FLOAT'
        },
        'primary_key': ['ELECTION_ID'],
        'description': 'Historical election summary with margins and trends'
    },

    # =========================================================================
    # CAMPAIGN FINANCE (MODIFIED - MULTI-RACE)
    # =========================================================================

    'CAMPAIGN_FINANCE_SUMMARY': {
        'source_key': 'finance_summary',
        'columns': {
            'FINANCE_ID': 'INTEGER AUTOINCREMENT',
            'ELECTION_YEAR': 'INTEGER',
            'RACE_TYPE': 'VARCHAR(50)',        # [ADD]
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',              # [ADD]
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'INCUMBENT': 'BOOLEAN',
            'COMMITTEE': 'VARCHAR(200)',
            'FEC_ID': 'VARCHAR(20)',            # [ADD] For federal races
            'TOTAL_RAISED': 'NUMBER(15,2)',
            'TOTAL_SPENT': 'NUMBER(15,2)',
            'CASH_ON_HAND': 'NUMBER(15,2)',
            'INDIVIDUAL_CONTRIBUTIONS': 'NUMBER(15,2)',
            'PAC_CONTRIBUTIONS': 'NUMBER(15,2)',
            'SMALL_DOLLAR_PCT': 'FLOAT',        # [ADD]
            'NUM_CONTRIBUTORS': 'INTEGER',
            'AVG_CONTRIBUTION': 'NUMBER(10,2)',
            'FUNDRAISING_EFFICIENCY': 'FLOAT',
            'BURN_RATE': 'FLOAT',
            'DATA_SOURCE': 'VARCHAR(100)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['FINANCE_ID'],
        'description': 'Campaign finance summary by candidate'
    },

    # =========================================================================
    # POLLING (MODIFIED - MULTI-RACE)
    # =========================================================================

    'POLLS': {
        'source_key': 'polls',
        'columns': {
            'POLL_ID': 'INTEGER AUTOINCREMENT',
            'RACE_TYPE': 'VARCHAR(50)',        # [ADD]
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',              # [ADD]
            'ELECTION_YEAR': 'INTEGER',
            'POLLSTER': 'VARCHAR(200)',
            'POLLSTER_RATING': 'VARCHAR(10)',   # [ADD] 538 rating
            'START_DATE': 'DATE',
            'END_DATE': 'DATE',
            'MID_DATE': 'DATE',
            'SAMPLE_SIZE': 'INTEGER',
            'POPULATION': 'VARCHAR(10)',        # LV, RV, A
            'MOE': 'FLOAT',
            'REPUBLICAN': 'FLOAT',
            'DEMOCRAT': 'FLOAT',
            'OTHER': 'FLOAT',
            'MARGIN': 'FLOAT',                  # R - D
            'REPUBLICAN_CANDIDATE': 'VARCHAR(100)',
            'DEMOCRAT_CANDIDATE': 'VARCHAR(100)',
            'DAYS_TO_ELECTION': 'INTEGER',
            'PARTISAN_SPONSOR': 'BOOLEAN',      # [ADD]
            'DATA_SOURCE': 'VARCHAR(100)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['POLL_ID'],
        'description': 'Individual poll results for all races'
    },

    'POLL_AVERAGES': {
        'source_key': 'poll_averages',
        'columns': {
            'AVERAGE_ID': 'INTEGER AUTOINCREMENT',
            'RACE_TYPE': 'VARCHAR(50)',        # [ADD]
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',              # [ADD]
            'ELECTION_YEAR': 'INTEGER',
            'PERIOD': 'VARCHAR(50)',            # final_week, final_month, etc.
            'NUM_POLLS': 'INTEGER',
            'AVG_REPUBLICAN': 'FLOAT',
            'AVG_DEMOCRAT': 'FLOAT',
            'AVG_MARGIN': 'FLOAT',
            'STD_MARGIN': 'FLOAT',
            'WEIGHTED_AVG_MARGIN': 'FLOAT',     # [ADD] Pollster quality weighted
            'ACTUAL_MARGIN': 'FLOAT',
            'POLLING_ERROR': 'FLOAT'
        },
        'primary_key': ['AVERAGE_ID'],
        'description': 'Polling averages by race and period'
    },

    # =========================================================================
    # NEWS & SENTIMENT (MODIFIED - MULTI-RACE)
    # =========================================================================

    'NEWS_ARTICLES': {
        'source_key': 'news_articles',
        'columns': {
            'ARTICLE_ID': 'INTEGER AUTOINCREMENT',
            'RACE_TYPE': 'VARCHAR(50)',        # [ADD]
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',              # [ADD]
            'ELECTION_YEAR': 'INTEGER',
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'SOURCE': 'VARCHAR(100)',
            'TITLE': 'VARCHAR(500)',
            'URL': 'VARCHAR(1000)',
            'PUBLISHED_DATE': 'DATE',
            'SNIPPET': 'VARCHAR(2000)',
            'WORD_COUNT': 'INTEGER',
            'SCOPE': 'VARCHAR(20)',             # national, state, local
            'TOPIC': 'VARCHAR(50)',
            'SENTIMENT_SCORE': 'FLOAT',         # [MODIFY] Continuous score
            'SENTIMENT_POSITIVE': 'FLOAT',      # [ADD]
            'SENTIMENT_NEGATIVE': 'FLOAT',      # [ADD]
            'SENTIMENT_NEUTRAL': 'FLOAT',       # [ADD]
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['ARTICLE_ID'],
        'description': 'News articles about candidates'
    },

    # =========================================================================
    # MODEL OUTPUTS (NEW)
    # =========================================================================

    'MODEL_PREDICTIONS': {
        'source_key': None,
        'columns': {
            'PREDICTION_ID': 'INTEGER AUTOINCREMENT',
            'MODEL_VERSION': 'VARCHAR(50)',
            'PREDICTION_DATE': 'TIMESTAMP_NTZ',
            'RACE_TYPE': 'VARCHAR(50)',
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',
            'ELECTION_YEAR': 'INTEGER',
            'MARGIN_ESTIMATE': 'FLOAT',         # Point estimate
            'MARGIN_CI_LOWER_95': 'FLOAT',      # 95% CI lower
            'MARGIN_CI_UPPER_95': 'FLOAT',      # 95% CI upper
            'MARGIN_CI_LOWER_80': 'FLOAT',      # 80% CI lower
            'MARGIN_CI_UPPER_80': 'FLOAT',      # 80% CI upper
            'STD_ERROR': 'FLOAT',
            'WIN_PROB_R': 'FLOAT',
            'WIN_PROB_D': 'FLOAT',
            'CONFIDENCE_TIER': 'INTEGER',       # 1=low, 2=medium, 3=high
            'IS_LOW_CONFIDENCE': 'BOOLEAN',
            'FEATURE_COMPLETENESS': 'FLOAT',
            'TOP_FACTORS': 'VARIANT',           # JSON of contributing factors
            'ACTUAL_MARGIN': 'FLOAT',           # Filled in post-election
            'PREDICTION_ERROR': 'FLOAT',        # Filled in post-election
            'NOTES': 'VARCHAR(500)'
        },
        'primary_key': ['PREDICTION_ID'],
        'description': 'Model predictions with uncertainty'
    },

    'TRAINING_DATA': {
        'source_key': None,
        'columns': {
            'TRAINING_ID': 'INTEGER AUTOINCREMENT',
            'RACE_TYPE': 'VARCHAR(50)',
            'STATE': 'VARCHAR(2)',
            'DISTRICT': 'INTEGER',
            'ELECTION_YEAR': 'INTEGER',
            'MARGIN_PCT': 'FLOAT',              # Target variable
            # Tier 1 features
            'PARTISAN_LEAN': 'FLOAT',
            'INCUMBENCY': 'INTEGER',
            'FUNDING_RATIO': 'FLOAT',
            'GDP_GROWTH': 'FLOAT',
            'UNEMPLOYMENT_RATE': 'FLOAT',
            'INFLATION_RATE': 'FLOAT',
            'NATIONAL_ENVIRONMENT': 'FLOAT',
            'PRESIDENTIAL_APPROVAL': 'FLOAT',
            'ELECTION_CONTEXT': 'VARCHAR(20)',
            'SAME_PARTY_AS_PRESIDENT': 'INTEGER',
            # Tier 2 features
            'POLL_MARGIN_MEAN': 'FLOAT',
            'POLL_MARGIN_STD': 'FLOAT',
            'POLL_COUNT': 'INTEGER',
            'VIX_MEAN': 'FLOAT',
            'NEWS_SENTIMENT_POSITIVE': 'FLOAT',
            'NEWS_SENTIMENT_NEGATIVE': 'FLOAT',
            # Tier 3 features
            'MIDTERM_PENALTY': 'FLOAT',
            'COATTAIL_EFFECT': 'FLOAT',
            'CULTURE_WAR_EXPOSURE': 'INTEGER',
            # Metadata
            'CONFIDENCE_TIER': 'INTEGER',
            'FEATURE_COMPLETENESS': 'FLOAT',
            'CREATED_DATE': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['TRAINING_ID'],
        'description': 'ML training data with all features'
    },

    'FEATURE_AVAILABILITY': {
        'source_key': None,
        'columns': {
            'AVAILABILITY_ID': 'INTEGER AUTOINCREMENT',
            'RACE_TYPE': 'VARCHAR(50)',
            'STATE': 'VARCHAR(2)',
            'ELECTION_YEAR': 'INTEGER',
            'HAS_POLLING': 'BOOLEAN',
            'POLL_COUNT': 'INTEGER',
            'HAS_FINANCE_DATA': 'BOOLEAN',
            'HAS_NEWS_SENTIMENT': 'BOOLEAN',
            'HAS_VIX_DATA': 'BOOLEAN',
            'HAS_CULTURE_WAR': 'BOOLEAN',
            'TIER1_COMPLETE': 'BOOLEAN',
            'TIER2_COMPLETE': 'BOOLEAN',
            'TIER3_COMPLETE': 'BOOLEAN',
            'OVERALL_COMPLETENESS': 'FLOAT',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['AVAILABILITY_ID'],
        'description': 'Tracks feature availability by race'
    },

    'MODEL_DIAGNOSTICS': {
        'source_key': None,
        'columns': {
            'DIAGNOSTIC_ID': 'INTEGER AUTOINCREMENT',
            'MODEL_VERSION': 'VARCHAR(50)',
            'TRAINING_DATE': 'TIMESTAMP_NTZ',
            'N_TRAINING_SAMPLES': 'INTEGER',
            'N_FEATURES': 'INTEGER',
            'RMSE': 'FLOAT',
            'MAE': 'FLOAT',
            'R2_SCORE': 'FLOAT',
            'LOO_SCORE': 'FLOAT',
            'RACE_TYPE_EFFECTS': 'VARIANT',     # JSON
            'TOP_STATE_EFFECTS': 'VARIANT',     # JSON
            'YEAR_EFFECTS': 'VARIANT',          # JSON
            'FEATURE_IMPORTANCE': 'VARIANT',    # JSON
            'CROSS_VAL_SCORES': 'VARIANT',      # JSON
            'NOTES': 'VARCHAR(500)'
        },
        'primary_key': ['DIAGNOSTIC_ID'],
        'description': 'Model training diagnostics'
    },

    # =========================================================================
    # MACROECONOMIC (KEEP - No race-specific changes needed)
    # =========================================================================

    'MACRO_NATIONAL_ENVIRONMENT': {
        'source_key': 'national_environment',
        'columns': {
            'DATE': 'DATE',
            'GENERIC_BALLOT_R': 'FLOAT',        # [ADD]
            'GENERIC_BALLOT_D': 'FLOAT',        # [ADD]
            'GENERIC_BALLOT_MARGIN': 'FLOAT',
            'PRESIDENTIAL_APPROVAL': 'FLOAT',
            'PRESIDENTIAL_DISAPPROVAL': 'FLOAT',
            'PRESIDENT_PARTY': 'VARCHAR(1)',
            'GDP_GROWTH': 'FLOAT',
            'UNEMPLOYMENT_RATE': 'FLOAT',
            'INFLATION_RATE': 'FLOAT',
            'CONSUMER_SENTIMENT': 'FLOAT',
            'VIX': 'FLOAT',
            'DATA_SOURCE': 'VARCHAR(100)'
        },
        'primary_key': ['DATE'],
        'description': 'National political and economic environment'
    },

    # [KEEP] Existing macro tables unchanged
    'MACRO_GDP': {
        'source_key': 'gdp',
        'columns': {
            'DATE': 'DATE',
            'GDP_NOMINAL': 'FLOAT',
            'GDP_REAL': 'FLOAT',
            'GDP_GROWTH_QOQ': 'FLOAT',
            'GDP_GROWTH_YOY': 'FLOAT'
        },
        'primary_key': ['DATE'],
        'description': 'GDP headline data'
    },

    'MACRO_UNEMPLOYMENT': {
        'source_key': 'unemployment',
        'columns': {
            'DATE': 'DATE',
            'UNEMPLOYMENT_RATE_U3': 'FLOAT',
            'UNEMPLOYMENT_RATE_U6': 'FLOAT',
            'UNEMPLOYED_LEVEL': 'INTEGER'
        },
        'primary_key': ['DATE'],
        'description': 'Unemployment measures'
    },

    'MACRO_CPI': {
        'source_key': 'macro_cpi',
        'columns': {
            'DATE': 'DATE',
            'CPI_ALL': 'FLOAT',
            'CPI_CORE': 'FLOAT',
            'CPI_YOY': 'FLOAT',
            'CPI_MOM': 'FLOAT'
        },
        'primary_key': ['DATE'],
        'description': 'Consumer Price Index data'
    },

    'VIX_DAILY': {
        'source_key': 'vix_daily',
        'columns': {
            'DATE': 'DATE',
            'VIX': 'FLOAT',
            'VIX_CHANGE': 'FLOAT',
            'VIX_PCT_CHANGE': 'FLOAT'
        },
        'primary_key': ['DATE'],
        'description': 'Daily VIX volatility index values'
    },

    # [KEEP] Culture war tables
    'CULTURE_WAR_EVENTS': {
        'source_key': 'culture_war_events',
        'columns': {
            'EVENT_ID': 'INTEGER AUTOINCREMENT',
            'COMPANY': 'VARCHAR(200)',
            'YEAR': 'INTEGER',
            'CULTURE_WAR_EVENT': 'VARCHAR(1000)',
            'EVENT_DATE': 'DATE',
            'INDUSTRY': 'VARCHAR(200)',
            'TICKER': 'VARCHAR(10)',
            'ESTIMATED_POLITICAL_LEANING': 'VARCHAR(50)'
        },
        'primary_key': ['EVENT_ID'],
        'description': 'Culture war events by company'
    },

    # =========================================================================
    # METADATA (KEEP)
    # =========================================================================

    'ETL_LOAD_LOG': {
        'source_key': None,
        'columns': {
            'LOAD_ID': 'INTEGER AUTOINCREMENT',
            'LOAD_TIMESTAMP': 'TIMESTAMP_NTZ',
            'TABLE_NAME': 'VARCHAR(100)',
            'ROWS_LOADED': 'INTEGER',
            'STATUS': 'VARCHAR(20)',
            'ERROR_MESSAGE': 'VARCHAR(1000)',
            'DURATION_SECONDS': 'FLOAT'
        },
        'primary_key': ['LOAD_ID'],
        'description': 'ETL load history and audit log'
    },

    'DATA_DICTIONARY': {
        'source_key': None,
        'columns': {
            'TABLE_NAME': 'VARCHAR(100)',
            'COLUMN_NAME': 'VARCHAR(100)',
            'DATA_TYPE': 'VARCHAR(50)',
            'DESCRIPTION': 'VARCHAR(500)',
            'SOURCE': 'VARCHAR(200)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['TABLE_NAME', 'COLUMN_NAME'],
        'description': 'Data dictionary metadata'
    }
}


# =============================================================================
# SEED DATA FOR DIMENSION TABLES
# =============================================================================

RACE_TYPE_SEED_DATA = [
    {'RACE_TYPE_ID': 1, 'RACE_TYPE': 'governor', 'RACE_LEVEL': 'state', 'CYCLE_LENGTH': 4, 'DESCRIPTION': 'State Governor'},
    {'RACE_TYPE_ID': 2, 'RACE_TYPE': 'us_senate', 'RACE_LEVEL': 'federal', 'CYCLE_LENGTH': 6, 'DESCRIPTION': 'US Senate'},
    {'RACE_TYPE_ID': 3, 'RACE_TYPE': 'us_house', 'RACE_LEVEL': 'federal', 'CYCLE_LENGTH': 2, 'DESCRIPTION': 'US House of Representatives'},
    {'RACE_TYPE_ID': 4, 'RACE_TYPE': 'president', 'RACE_LEVEL': 'federal', 'CYCLE_LENGTH': 4, 'DESCRIPTION': 'President'},
    {'RACE_TYPE_ID': 5, 'RACE_TYPE': 'state_senate', 'RACE_LEVEL': 'state', 'CYCLE_LENGTH': 4, 'DESCRIPTION': 'State Senate'},
    {'RACE_TYPE_ID': 6, 'RACE_TYPE': 'state_house', 'RACE_LEVEL': 'state', 'CYCLE_LENGTH': 2, 'DESCRIPTION': 'State House'},
]

STATE_PARTISAN_LEAN_DATA = {
    'AL': 14.8, 'AK': 9.1, 'AZ': 0.3, 'AR': 16.6, 'CA': -14.3,
    'CO': -3.9, 'CT': -6.5, 'DE': -6.6, 'FL': 2.8, 'GA': 0.5,
    'HI': -15.0, 'ID': 19.1, 'IL': -7.2, 'IN': 10.8, 'IA': 5.8,
    'KS': 11.2, 'KY': 15.5, 'LA': 11.6, 'ME': -2.6, 'MD': -13.5,
    'MA': -14.5, 'MI': -0.5, 'MN': -1.4, 'MS': 9.6, 'MO': 10.4,
    'MT': 10.1, 'NE': 12.0, 'NV': -1.3, 'NH': -0.3, 'NJ': -6.4,
    'NM': -5.4, 'NY': -10.0, 'NC': 1.8, 'ND': 19.9, 'OH': 5.8,
    'OK': 20.0, 'OR': -5.8, 'PA': -0.2, 'RI': -10.3, 'SC': 7.9,
    'SD': 15.4, 'TN': 14.3, 'TX': 5.5, 'UT': 11.4, 'VT': -15.7,
    'VA': -3.2, 'WA': -8.4, 'WV': 19.3, 'WI': -0.2, 'WY': 25.6,
    'DC': -43.0
}


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def generate_migration_sql() -> str:
    """
    Generate SQL to migrate from TEXAS_GOVERNOR schema to ELECTION_PREDICTIONS.

    Returns DDL and DML statements for migration.
    """
    migration_sql = f"""
-- =============================================================================
-- MIGRATION: TEXAS_GOVERNOR → ELECTION_PREDICTIONS
-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-- =============================================================================

-- Step 1: Create new schema
CREATE SCHEMA IF NOT EXISTS {DEFAULT_SCHEMA};
USE SCHEMA {DEFAULT_SCHEMA};

-- Step 2: Create dimension tables
CREATE TABLE IF NOT EXISTS DIM_RACE_TYPE (
    RACE_TYPE_ID INTEGER,
    RACE_TYPE VARCHAR(50),
    RACE_LEVEL VARCHAR(20),
    CYCLE_LENGTH INTEGER,
    DESCRIPTION VARCHAR(200),
    PRIMARY KEY (RACE_TYPE_ID)
);

-- Seed race type data
INSERT INTO DIM_RACE_TYPE VALUES
    (1, 'governor', 'state', 4, 'State Governor'),
    (2, 'us_senate', 'federal', 6, 'US Senate'),
    (3, 'us_house', 'federal', 2, 'US House of Representatives'),
    (4, 'president', 'federal', 4, 'President');

-- Step 3: Migrate election results
INSERT INTO ELECTION_RESULTS (
    ELECTION_YEAR, ELECTION_DATE, RACE_TYPE, STATE, DISTRICT,
    CANDIDATE, PARTY, VOTES, VOTE_PERCENTAGE, TOTAL_VOTES,
    TURNOUT_PERCENTAGE, WINNER, INCUMBENT, DATA_SOURCE
)
SELECT
    ELECTION_YEAR, ELECTION_DATE, 'governor' AS RACE_TYPE, STATE, NULL AS DISTRICT,
    CANDIDATE, PARTY, VOTES, VOTE_PERCENTAGE, TOTAL_VOTES,
    TURNOUT_PERCENTAGE, WINNER, INCUMBENT, DATA_SOURCE
FROM TEXAS_GOVERNOR.ELECTION_RESULTS_STATEWIDE;

-- Step 4: Migrate polling data
INSERT INTO POLLS (
    RACE_TYPE, STATE, DISTRICT, ELECTION_YEAR, POLLSTER, START_DATE, END_DATE,
    MID_DATE, SAMPLE_SIZE, POPULATION, MOE, REPUBLICAN, DEMOCRAT, OTHER, MARGIN,
    REPUBLICAN_CANDIDATE, DEMOCRAT_CANDIDATE, DAYS_TO_ELECTION, DATA_SOURCE
)
SELECT
    'governor', STATE, NULL, ELECTION_YEAR, POLLSTER, START_DATE, END_DATE,
    MID_DATE, SAMPLE_SIZE, POPULATION, MOE, REPUBLICAN, DEMOCRAT, OTHER, MARGIN,
    REPUBLICAN_CANDIDATE, DEMOCRAT_CANDIDATE, DAYS_TO_ELECTION, DATA_SOURCE
FROM TEXAS_GOVERNOR.POLLS;

-- Step 5: Migrate campaign finance
INSERT INTO CAMPAIGN_FINANCE_SUMMARY (
    ELECTION_YEAR, RACE_TYPE, STATE, DISTRICT, CANDIDATE, PARTY, INCUMBENT,
    COMMITTEE, TOTAL_RAISED, TOTAL_SPENT, CASH_ON_HAND, INDIVIDUAL_CONTRIBUTIONS,
    PAC_CONTRIBUTIONS, NUM_CONTRIBUTORS, AVG_CONTRIBUTION, DATA_SOURCE
)
SELECT
    ELECTION_YEAR, 'governor', 'TX', NULL, CANDIDATE, PARTY, INCUMBENT,
    COMMITTEE, TOTAL_RAISED, TOTAL_SPENT, CASH_ON_HAND, INDIVIDUAL_CONTRIBUTIONS,
    PAC_CONTRIBUTIONS, NUM_CONTRIBUTORS, AVG_CONTRIBUTION, DATA_SOURCE
FROM TEXAS_GOVERNOR.CAMPAIGN_FINANCE_SUMMARY;

-- Step 6: Create views for backward compatibility
CREATE OR REPLACE VIEW TEXAS_GOVERNOR_RESULTS AS
SELECT * FROM ELECTION_RESULTS WHERE STATE = 'TX' AND RACE_TYPE = 'governor';

CREATE OR REPLACE VIEW TEXAS_GOVERNOR_POLLS AS
SELECT * FROM POLLS WHERE STATE = 'TX' AND RACE_TYPE = 'governor';

-- Migration complete
"""
    return migration_sql


def generate_ddl_v2() -> str:
    """Generate complete DDL for v2 schema."""
    ddl_lines = [
        "-- Snowflake DDL for Election Predictions (Multi-Race)",
        f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"CREATE SCHEMA IF NOT EXISTS {DEFAULT_SCHEMA};",
        f"USE SCHEMA {DEFAULT_SCHEMA};",
        ""
    ]

    for table_name, table_def in SNOWFLAKE_TABLES_V2.items():
        columns = table_def['columns']
        primary_key = table_def.get('primary_key', [])

        col_defs = []
        for col_name, col_type in columns.items():
            col_defs.append(f"    {col_name} {col_type}")

        pk_cols = [pk for pk in primary_key if 'AUTOINCREMENT' not in columns.get(pk, '')]
        if pk_cols:
            col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")

        ddl_lines.append(f"-- {table_def.get('description', '')}")
        ddl_lines.append(f"CREATE TABLE IF NOT EXISTS {table_name} (")
        ddl_lines.append(',\n'.join(col_defs))
        ddl_lines.append(");")
        ddl_lines.append("")

    return '\n'.join(ddl_lines)


# =============================================================================
# DATABASE MANAGER V2
# =============================================================================

class DatabaseManagerV2:
    """
    Database manager for multi-race election prediction schema.

    REPLACES: DatabaseManager from database.py
    """

    def __init__(self, schema: str = DEFAULT_SCHEMA, **kwargs):
        self.schema = schema
        self.connection = None
        self.cursor = None

        # Connection params from environment
        self.account = kwargs.get('account') or os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = kwargs.get('user') or os.getenv('SNOWFLAKE_USER')
        self.password = kwargs.get('password') or os.getenv('SNOWFLAKE_PASSWORD')
        self.warehouse = kwargs.get('warehouse') or os.getenv('SNOWFLAKE_WAREHOUSE', DEFAULT_WAREHOUSE)
        self.database = kwargs.get('database') or os.getenv('SNOWFLAKE_DATABASE')

    def connect(self) -> bool:
        """Establish Snowflake connection."""
        if not SNOWFLAKE_AVAILABLE:
            logger.error("Snowflake connector not available")
            return False

        try:
            self.connection = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            self.cursor = self.connection.cursor()
            logger.info(f"Connected to Snowflake: {self.database}.{self.schema}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Close connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def setup_schema(self, drop_existing: bool = False) -> Dict[str, bool]:
        """Create schema and all tables."""
        results = {}

        # Create schema
        self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        self.cursor.execute(f"USE SCHEMA {self.schema}")
        results['schema'] = True

        # Create tables
        for table_name, table_def in SNOWFLAKE_TABLES_V2.items():
            try:
                if drop_existing:
                    self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

                columns = table_def['columns']
                col_defs = [f"{name} {dtype}" for name, dtype in columns.items()]

                pk_cols = [pk for pk in table_def.get('primary_key', [])
                          if 'AUTOINCREMENT' not in columns.get(pk, '')]
                if pk_cols:
                    col_defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")

                sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
                self.cursor.execute(sql)
                results[table_name] = True
                logger.info(f"Created table: {table_name}")

            except Exception as e:
                results[table_name] = False
                logger.error(f"Failed to create {table_name}: {e}")

        return results

    def store_prediction(self, prediction: Dict, race_info: Dict) -> bool:
        """
        Store a model prediction.

        Args:
            prediction: PredictionResult.to_dict() output
            race_info: Dict with race_type, state, district, election_year
        """
        try:
            sql = """
            INSERT INTO MODEL_PREDICTIONS (
                MODEL_VERSION, PREDICTION_DATE, RACE_TYPE, STATE, DISTRICT, ELECTION_YEAR,
                MARGIN_ESTIMATE, MARGIN_CI_LOWER_95, MARGIN_CI_UPPER_95,
                MARGIN_CI_LOWER_80, MARGIN_CI_UPPER_80, STD_ERROR,
                WIN_PROB_R, WIN_PROB_D, CONFIDENCE_TIER, IS_LOW_CONFIDENCE,
                FEATURE_COMPLETENESS, TOP_FACTORS
            ) VALUES (
                %s, CURRENT_TIMESTAMP(), %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """

            params = (
                'v2.0',
                race_info['race_type'],
                race_info['state'],
                race_info.get('district'),
                race_info['election_year'],
                prediction['margin_estimate'],
                prediction['margin_ci_95'][0],
                prediction['margin_ci_95'][1],
                prediction['margin_ci_80'][0],
                prediction['margin_ci_80'][1],
                prediction['std_error'],
                prediction['win_prob_r'],
                prediction['win_prob_d'],
                prediction['confidence_tier'],
                prediction['is_low_confidence'],
                prediction['feature_completeness'],
                json.dumps(prediction['top_factors'])
            )

            self.cursor.execute(sql, params)
            return True

        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            return False

    def get_training_data(
        self,
        race_types: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        min_year: int = 2010,
        max_year: int = 2024
    ) -> pd.DataFrame:
        """
        Retrieve training data for model fitting.

        Args:
            race_types: List of race types to include
            states: List of states to include
            min_year: Minimum election year
            max_year: Maximum election year

        Returns:
            DataFrame with training features and targets
        """
        conditions = [f"ELECTION_YEAR >= {min_year}", f"ELECTION_YEAR <= {max_year}"]

        if race_types:
            race_list = ",".join([f"'{r}'" for r in race_types])
            conditions.append(f"RACE_TYPE IN ({race_list})")

        if states:
            state_list = ",".join([f"'{s}'" for s in states])
            conditions.append(f"STATE IN ({state_list})")

        where_clause = " AND ".join(conditions)

        sql = f"""
        SELECT * FROM TRAINING_DATA
        WHERE {where_clause}
        ORDER BY ELECTION_YEAR, STATE, RACE_TYPE
        """

        self.cursor.execute(sql)
        columns = [desc[0] for desc in self.cursor.description]
        data = self.cursor.fetchall()

        return pd.DataFrame(data, columns=columns)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Database V2 for Multi-Race Elections')

    parser.add_argument('--setup', action='store_true', help='Create schema and tables')
    parser.add_argument('--migrate', action='store_true', help='Migrate from TEXAS_GOVERNOR')
    parser.add_argument('--generate-ddl', action='store_true', help='Generate DDL file')
    parser.add_argument('--drop-existing', action='store_true', help='Drop existing tables')

    args = parser.parse_args()

    if args.generate_ddl:
        ddl = generate_ddl_v2()
        with open('snowflake_ddl_v2.sql', 'w') as f:
            f.write(ddl)
        print("DDL written to snowflake_ddl_v2.sql")
        print(ddl)

    elif args.migrate:
        migration = generate_migration_sql()
        with open('migration_to_v2.sql', 'w') as f:
            f.write(migration)
        print("Migration SQL written to migration_to_v2.sql")
        print(migration)

    elif args.setup:
        db = DatabaseManagerV2()
        if db.connect():
            results = db.setup_schema(drop_existing=args.drop_existing)
            print(f"Setup complete: {sum(results.values())}/{len(results)} successful")
            db.disconnect()
        else:
            print("Failed to connect to Snowflake")

    else:
        parser.print_help()
