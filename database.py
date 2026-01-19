"""
Snowflake Database Module for Texas Governor Race Political Data.

This module provides functionality to load ETL-processed data into Snowflake,
including schema creation, table management, and bulk data loading.

Usage:
    python database.py --setup              # Create schemas and tables
    python database.py --load               # Load data from ETL
    python database.py --full               # Setup + Load
    python database.py --query "SELECT..."  # Run ad-hoc query

Prerequisites:
    pip install snowflake-connector-python snowflake-sqlalchemy pandas

Environment Variables Required:
    SNOWFLAKE_ACCOUNT   - Snowflake account identifier
    SNOWFLAKE_USER      - Snowflake username
    SNOWFLAKE_PASSWORD  - Snowflake password
    SNOWFLAKE_WAREHOUSE - Snowflake warehouse name
    SNOWFLAKE_DATABASE  - Snowflake database name
    SNOWFLAKE_SCHEMA    - Snowflake schema name (default: TEXAS_GOVERNOR)
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

# Snowflake imports
try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("Warning: snowflake-connector-python not installed. Install with:")
    print("  pip install snowflake-connector-python")

# Import from project modules
from ETL import (
    ETLPipeline,
    DATA_DICTIONARY,
    get_data_dictionary,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR
)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default Snowflake configuration
DEFAULT_SCHEMA = 'TEXAS_GOVERNOR'
DEFAULT_WAREHOUSE = 'COMPUTE_WH'

# =============================================================================
# DATA DICTIONARY TO SNOWFLAKE SCHEMA MAPPING
# =============================================================================
"""
Maps the data dictionary from ETL.py to Snowflake table definitions.

Snowflake Schema: TEXAS_GOVERNOR
Tables:
    - ELECTION_RESULTS_STATEWIDE
    - ELECTION_RESULTS_COUNTY
    - ELECTION_HISTORICAL
    - CAMPAIGN_FINANCE_CONTRIBUTIONS
    - CAMPAIGN_FINANCE_EXPENDITURES
    - CAMPAIGN_FINANCE_SUMMARY
    - CAMPAIGN_FINANCE_DONORS
    - POLLS
    - POLL_AVERAGES
    - POLLSTERS
    - POLL_TRENDS
    - NEWS_ARTICLES
    - NEWS_BY_CANDIDATE
    - NEWS_BY_SOURCE
    - NEWS_BY_TOPIC
    - NEWS_TIMELINE
    - NEWS_COVERAGE_SUMMARY
    - CANDIDATE_MASTER
    - ELECTION_CYCLE_SUMMARY
    - ETL_TIMELINE
"""

# Snowflake table definitions with column types
SNOWFLAKE_TABLES = {
    # Election Tables
    'ELECTION_RESULTS_STATEWIDE': {
        'source_key': 'election_results',
        'columns': {
            'ELECTION_YEAR': 'INTEGER',
            'ELECTION_DATE': 'DATE',
            'RACE': 'VARCHAR(50)',
            'STATE': 'VARCHAR(2)',
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'VOTES': 'INTEGER',
            'VOTE_PERCENTAGE': 'FLOAT',
            'TOTAL_VOTES': 'INTEGER',
            'TURNOUT_PERCENTAGE': 'FLOAT',
            'WINNER': 'BOOLEAN',
            'INCUMBENT': 'BOOLEAN',
            'DATA_SOURCE': 'VARCHAR(100)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['ELECTION_YEAR', 'CANDIDATE'],
        'description': 'Statewide election results by candidate'
    },
    'ELECTION_HISTORICAL': {
        'source_key': 'election_historical',
        'columns': {
            'ELECTION_YEAR': 'INTEGER',
            'ELECTION_DATE': 'DATE',
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
            'MARGIN_CHANGE': 'FLOAT',
            'TURNOUT_CHANGE': 'FLOAT'
        },
        'primary_key': ['ELECTION_YEAR'],
        'description': 'Historical election summary with margins and trends'
    },

    # Campaign Finance Tables
    'CAMPAIGN_FINANCE_SUMMARY': {
        'source_key': 'finance_summary',
        'columns': {
            'ELECTION_YEAR': 'INTEGER',
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'INCUMBENT': 'BOOLEAN',
            'COMMITTEE': 'VARCHAR(200)',
            'TOTAL_RAISED': 'NUMBER(15,2)',
            'TOTAL_SPENT': 'NUMBER(15,2)',
            'CASH_ON_HAND': 'NUMBER(15,2)',
            'INDIVIDUAL_CONTRIBUTIONS': 'NUMBER(15,2)',
            'PAC_CONTRIBUTIONS': 'NUMBER(15,2)',
            'OTHER_CONTRIBUTIONS': 'NUMBER(15,2)',
            'NUM_CONTRIBUTORS': 'INTEGER',
            'AVG_CONTRIBUTION': 'NUMBER(10,2)',
            'FUNDRAISING_EFFICIENCY': 'FLOAT',
            'BURN_RATE': 'FLOAT',
            'DATA_SOURCE': 'VARCHAR(100)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['ELECTION_YEAR', 'CANDIDATE'],
        'description': 'Campaign finance summary by candidate'
    },
    'CAMPAIGN_FINANCE_CONTRIBUTIONS': {
        'source_key': 'finance_contributions',
        'columns': {
            'CONTRIBUTION_ID': 'INTEGER AUTOINCREMENT',
            'ELECTION_YEAR': 'INTEGER',
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'DONOR_NAME': 'VARCHAR(200)',
            'AMOUNT': 'NUMBER(12,2)',
            'DONOR_EMPLOYER': 'VARCHAR(200)',
            'CONTRIBUTION_DATE': 'DATE',
            'CONTRIBUTION_TYPE': 'VARCHAR(50)',
            'SOURCE': 'VARCHAR(100)'
        },
        'primary_key': ['CONTRIBUTION_ID'],
        'description': 'Individual campaign contributions'
    },
    'CAMPAIGN_FINANCE_EXPENDITURES': {
        'source_key': 'finance_expenditures',
        'columns': {
            'EXPENDITURE_ID': 'INTEGER AUTOINCREMENT',
            'ELECTION_YEAR': 'INTEGER',
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'COMMITTEE': 'VARCHAR(200)',
            'CATEGORY': 'VARCHAR(100)',
            'AMOUNT': 'NUMBER(15,2)',
            'PERCENTAGE': 'FLOAT',
            'EXPENDITURE_DATE': 'DATE',
            'SOURCE': 'VARCHAR(100)'
        },
        'primary_key': ['EXPENDITURE_ID'],
        'description': 'Campaign expenditures by category'
    },

    # Polling Tables
    'POLLS': {
        'source_key': 'polls',
        'columns': {
            'POLL_ID': 'INTEGER AUTOINCREMENT',
            'POLLSTER': 'VARCHAR(200)',
            'START_DATE': 'DATE',
            'END_DATE': 'DATE',
            'MID_DATE': 'DATE',
            'SAMPLE_SIZE': 'INTEGER',
            'POPULATION': 'VARCHAR(10)',
            'MOE': 'FLOAT',
            'REPUBLICAN': 'FLOAT',
            'DEMOCRAT': 'FLOAT',
            'OTHER': 'FLOAT',
            'MARGIN': 'FLOAT',
            'REPUBLICAN_CANDIDATE': 'VARCHAR(100)',
            'DEMOCRAT_CANDIDATE': 'VARCHAR(100)',
            'ELECTION_YEAR': 'INTEGER',
            'RACE': 'VARCHAR(50)',
            'STATE': 'VARCHAR(2)',
            'DAYS_TO_ELECTION': 'INTEGER',
            'DATA_SOURCE': 'VARCHAR(100)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['POLL_ID'],
        'description': 'Individual poll results'
    },
    'POLL_AVERAGES': {
        'source_key': 'poll_averages',
        'columns': {
            'ELECTION_YEAR': 'INTEGER',
            'PERIOD': 'VARCHAR(50)',
            'REPUBLICAN_CANDIDATE': 'VARCHAR(100)',
            'DEMOCRAT_CANDIDATE': 'VARCHAR(100)',
            'NUM_POLLS': 'INTEGER',
            'AVG_REPUBLICAN': 'FLOAT',
            'AVG_DEMOCRAT': 'FLOAT',
            'AVG_MARGIN': 'FLOAT',
            'MIN_MARGIN': 'FLOAT',
            'MAX_MARGIN': 'FLOAT',
            'STD_MARGIN': 'FLOAT',
            'ACTUAL_REPUBLICAN': 'FLOAT',
            'ACTUAL_DEMOCRAT': 'FLOAT',
            'ACTUAL_MARGIN': 'FLOAT',
            'POLLING_ERROR': 'FLOAT'
        },
        'primary_key': ['ELECTION_YEAR', 'PERIOD'],
        'description': 'Polling averages by election cycle'
    },
    'POLLSTERS': {
        'source_key': 'pollsters',
        'columns': {
            'POLLSTER': 'VARCHAR(200)',
            'TYPE': 'VARCHAR(50)',
            'METHODOLOGY': 'VARCHAR(100)',
            'FIVETHIRTYEIGHT_RATING': 'VARCHAR(10)',
            'PARTISAN_LEAN': 'VARCHAR(20)',
            'TRANSPARENCY': 'VARCHAR(20)',
            'TYPICAL_SAMPLE': 'INTEGER'
        },
        'primary_key': ['POLLSTER'],
        'description': 'Pollster ratings and methodology'
    },
    'POLL_TRENDS': {
        'source_key': 'poll_trends',
        'columns': {
            'ELECTION_YEAR': 'INTEGER',
            'REPUBLICAN_CANDIDATE': 'VARCHAR(100)',
            'DEMOCRAT_CANDIDATE': 'VARCHAR(100)',
            'INITIAL_MARGIN': 'FLOAT',
            'FINAL_MARGIN': 'FLOAT',
            'MARGIN_CHANGE': 'FLOAT',
            'PEAK_REPUBLICAN_LEAD': 'FLOAT',
            'MIN_REPUBLICAN_LEAD': 'FLOAT',
            'TREND_DIRECTION': 'VARCHAR(50)',
            'VOLATILITY': 'FLOAT',
            'NUM_POLLS': 'INTEGER'
        },
        'primary_key': ['ELECTION_YEAR'],
        'description': 'Polling trends by election cycle'
    },

    # News Tables
    'NEWS_ARTICLES': {
        'source_key': 'news_articles',
        'columns': {
            'ARTICLE_ID': 'INTEGER AUTOINCREMENT',
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'ELECTION_YEAR': 'INTEGER',
            'SOURCE': 'VARCHAR(100)',
            'TITLE': 'VARCHAR(500)',
            'URL': 'VARCHAR(1000)',
            'PUBLISHED_DATE': 'DATE',
            'SNIPPET': 'VARCHAR(2000)',
            'AUTHOR': 'VARCHAR(200)',
            'SECTION': 'VARCHAR(100)',
            'WORD_COUNT': 'INTEGER',
            'SCOPE': 'VARCHAR(20)',
            'TOPIC': 'VARCHAR(50)',
            'SENTIMENT': 'VARCHAR(20)',
            'LAST_UPDATED': 'TIMESTAMP_NTZ'
        },
        'primary_key': ['ARTICLE_ID'],
        'description': 'News articles about governor candidates'
    },
    'NEWS_COVERAGE_SUMMARY': {
        'source_key': 'news_coverage_summary',
        'columns': {
            'ELECTION_YEAR': 'INTEGER',
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'TOTAL_ARTICLES': 'INTEGER',
            'TEXAS_COVERAGE': 'INTEGER',
            'NATIONAL_COVERAGE': 'INTEGER',
            'UNIQUE_SOURCES': 'INTEGER',
            'TOP_TOPIC': 'VARCHAR(50)',
            'AVG_WORD_COUNT': 'FLOAT',
            'COVERAGE_RATIO': 'FLOAT'
        },
        'primary_key': ['ELECTION_YEAR', 'CANDIDATE'],
        'description': 'News coverage summary by candidate'
    },
    'NEWS_BY_TOPIC': {
        'source_key': 'news_by_topic',
        'columns': {
            'TOPIC': 'VARCHAR(50)',
            'ELECTION_YEAR': 'INTEGER',
            'ARTICLE_COUNT': 'INTEGER',
            'BY_CANDIDATE': 'VARIANT'
        },
        'primary_key': ['TOPIC', 'ELECTION_YEAR'],
        'description': 'News coverage by topic'
    },

    # Integrated Tables (from ETL transformation)
    'CANDIDATE_MASTER': {
        'source_key': 'candidate_master',
        'columns': {
            'CANDIDATE': 'VARCHAR(100)',
            'PARTY': 'VARCHAR(10)',
            'ELECTION_YEAR': 'INTEGER',
            'WON': 'BOOLEAN',
            'TOTAL_RAISED': 'NUMBER(15,2)',
            'TOTAL_SPENT': 'NUMBER(15,2)',
            'NUM_CONTRIBUTORS': 'INTEGER',
            'AVG_MARGIN': 'FLOAT',
            'POLLING_ERROR': 'FLOAT',
            'TOTAL_ARTICLES': 'INTEGER',
            'TOP_TOPIC': 'VARCHAR(50)'
        },
        'primary_key': ['CANDIDATE', 'ELECTION_YEAR'],
        'description': 'Master candidate dataset with all metrics'
    },
    'ELECTION_CYCLE_SUMMARY': {
        'source_key': 'election_cycle_summary',
        'columns': {
            'ELECTION_YEAR': 'INTEGER',
            'WINNER': 'VARCHAR(100)',
            'WINNER_PARTY': 'VARCHAR(10)',
            'MARGIN_PCT': 'FLOAT',
            'TOTAL_VOTES': 'INTEGER',
            'TURNOUT_PCT': 'FLOAT',
            'TOTAL_RAISED_ALL': 'NUMBER(15,2)',
            'TOTAL_SPENT_ALL': 'NUMBER(15,2)',
            'R_FUNDRAISING_ADVANTAGE': 'NUMBER(15,2)',
            'POLL_AVG_MARGIN': 'FLOAT',
            'POLLING_ERROR': 'FLOAT',
            'NUM_POLLS': 'INTEGER',
            'TOTAL_NEWS_ARTICLES': 'INTEGER'
        },
        'primary_key': ['ELECTION_YEAR'],
        'description': 'Summary metrics by election cycle'
    },
    'ETL_TIMELINE': {
        'source_key': 'timeline',
        'columns': {
            'EVENT_ID': 'INTEGER AUTOINCREMENT',
            'DATE': 'DATE',
            'ELECTION_YEAR': 'INTEGER',
            'EVENT_TYPE': 'VARCHAR(50)',
            'SOURCE': 'VARCHAR(200)',
            'VALUE': 'FLOAT',
            'DESCRIPTION': 'VARCHAR(500)'
        },
        'primary_key': ['EVENT_ID'],
        'description': 'Unified timeline of all events'
    },

    # Metadata Tables
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
# SNOWFLAKE CONNECTION
# =============================================================================
class SnowflakeConnection:
    """
    Manages Snowflake database connections.
    """

    def __init__(
        self,
        account: str = None,
        user: str = None,
        password: str = None,
        warehouse: str = None,
        database: str = None,
        schema: str = None
    ):
        """
        Initialize Snowflake connection parameters.

        Args:
            account: Snowflake account identifier
            user: Snowflake username
            password: Snowflake password
            warehouse: Snowflake warehouse name
            database: Snowflake database name
            schema: Snowflake schema name
        """
        self.account = account or os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = user or os.getenv('SNOWFLAKE_USER')
        self.password = password or os.getenv('SNOWFLAKE_PASSWORD')
        self.warehouse = warehouse or os.getenv('SNOWFLAKE_WAREHOUSE', DEFAULT_WAREHOUSE)
        self.database = database or os.getenv('SNOWFLAKE_DATABASE')
        self.schema = schema or os.getenv('SNOWFLAKE_SCHEMA', DEFAULT_SCHEMA)

        self.connection = None
        self.cursor = None

    def connect(self) -> bool:
        """
        Establish connection to Snowflake.

        Returns:
            True if connection successful, False otherwise
        """
        if not SNOWFLAKE_AVAILABLE:
            logger.error("Snowflake connector not available")
            return False

        if not all([self.account, self.user, self.password, self.database]):
            logger.error("Missing required Snowflake credentials")
            logger.error("Required environment variables:")
            logger.error("  SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_DATABASE")
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
            logger.error(f"Failed to connect to Snowflake: {e}")
            return False

    def disconnect(self) -> None:
        """Close Snowflake connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Snowflake")

    def execute(self, sql: str, params: tuple = None) -> Any:
        """
        Execute SQL statement.

        Args:
            sql: SQL statement to execute
            params: Optional parameters for parameterized query

        Returns:
            Cursor result
        """
        try:
            if params:
                return self.cursor.execute(sql, params)
            return self.cursor.execute(sql)
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            logger.error(f"SQL: {sql[:500]}...")
            raise

    def fetchall(self) -> List[tuple]:
        """Fetch all results from last query."""
        return self.cursor.fetchall()

    def fetchone(self) -> tuple:
        """Fetch one result from last query."""
        return self.cursor.fetchone()


# =============================================================================
# SCHEMA MANAGER
# =============================================================================
class SchemaManager:
    """
    Manages Snowflake schema and table creation.
    """

    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize Schema Manager.

        Args:
            connection: Active SnowflakeConnection
        """
        self.conn = connection

    def create_schema(self) -> bool:
        """
        Create the schema if it doesn't exist.

        Returns:
            True if successful
        """
        try:
            sql = f"CREATE SCHEMA IF NOT EXISTS {self.conn.schema}"
            self.conn.execute(sql)
            logger.info(f"Schema {self.conn.schema} ready")
            return True
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False

    def create_all_tables(self, drop_existing: bool = False) -> Dict[str, bool]:
        """
        Create all tables defined in SNOWFLAKE_TABLES.

        Args:
            drop_existing: If True, drop existing tables first

        Returns:
            Dictionary of table names and creation status
        """
        results = {}

        for table_name, table_def in SNOWFLAKE_TABLES.items():
            try:
                if drop_existing:
                    self.drop_table(table_name)

                success = self.create_table(table_name, table_def)
                results[table_name] = success

            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
                results[table_name] = False

        return results

    def create_table(self, table_name: str, table_def: Dict) -> bool:
        """
        Create a single table.

        Args:
            table_name: Name of table to create
            table_def: Table definition dictionary

        Returns:
            True if successful
        """
        columns = table_def['columns']
        primary_key = table_def.get('primary_key', [])

        # Build column definitions
        col_defs = []
        for col_name, col_type in columns.items():
            col_defs.append(f"    {col_name} {col_type}")

        # Add primary key constraint if not using AUTOINCREMENT
        pk_cols = [pk for pk in primary_key if 'AUTOINCREMENT' not in columns.get(pk, '')]
        if pk_cols:
            col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")

        sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
{','.join(col_defs)}
)
COMMENT = '{table_def.get('description', '')}'
"""

        try:
            self.conn.execute(sql)
            logger.info(f"  Created table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"  Failed to create table {table_name}: {e}")
            return False

    def drop_table(self, table_name: str) -> bool:
        """
        Drop a table if it exists.

        Args:
            table_name: Name of table to drop

        Returns:
            True if successful
        """
        try:
            sql = f"DROP TABLE IF EXISTS {table_name}"
            self.conn.execute(sql)
            logger.info(f"  Dropped table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"  Failed to drop table {table_name}: {e}")
            return False

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists
        """
        sql = f"""
SELECT COUNT(*)
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = '{self.conn.schema}'
AND TABLE_NAME = '{table_name}'
"""
        self.conn.execute(sql)
        result = self.conn.fetchone()
        return result[0] > 0

    def get_table_row_count(self, table_name: str) -> int:
        """
        Get row count for a table.

        Args:
            table_name: Name of table

        Returns:
            Number of rows
        """
        try:
            sql = f"SELECT COUNT(*) FROM {table_name}"
            self.conn.execute(sql)
            result = self.conn.fetchone()
            return result[0]
        except:
            return 0


# =============================================================================
# DATA LOADER
# =============================================================================
class DataLoader:
    """
    Loads data from ETL pipeline into Snowflake.
    """

    def __init__(self, connection: SnowflakeConnection):
        """
        Initialize Data Loader.

        Args:
            connection: Active SnowflakeConnection
        """
        self.conn = connection
        self.load_log = []

    def load_all_data(
        self,
        transformed_data: Dict[str, pd.DataFrame],
        truncate_first: bool = True
    ) -> Dict[str, Dict]:
        """
        Load all transformed data into Snowflake tables.

        Args:
            transformed_data: Dictionary of DataFrames from ETL
            truncate_first: If True, truncate tables before loading

        Returns:
            Dictionary of load results
        """
        results = {}

        logger.info("Starting data load to Snowflake...")

        for table_name, table_def in SNOWFLAKE_TABLES.items():
            source_key = table_def.get('source_key')

            if source_key is None:
                # Skip metadata tables
                continue

            if source_key not in transformed_data:
                logger.warning(f"  No data for {table_name} (source: {source_key})")
                results[table_name] = {'status': 'skipped', 'reason': 'no source data'}
                continue

            df = transformed_data[source_key]

            if df is None or df.empty:
                logger.warning(f"  Empty DataFrame for {table_name}")
                results[table_name] = {'status': 'skipped', 'reason': 'empty dataframe'}
                continue

            result = self.load_table(table_name, df, table_def, truncate_first)
            results[table_name] = result

        # Load data dictionary
        self._load_data_dictionary()

        return results

    def load_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        table_def: Dict,
        truncate_first: bool = True
    ) -> Dict:
        """
        Load a single DataFrame into a Snowflake table.

        Args:
            table_name: Target table name
            df: DataFrame to load
            table_def: Table definition
            truncate_first: If True, truncate table before loading

        Returns:
            Load result dictionary
        """
        start_time = datetime.now()

        try:
            # Prepare DataFrame
            df_prepared = self._prepare_dataframe(df, table_def)

            # Truncate if requested
            if truncate_first:
                self.conn.execute(f"TRUNCATE TABLE IF EXISTS {table_name}")

            # Use write_pandas for efficient loading
            success, num_chunks, num_rows, output = write_pandas(
                conn=self.conn.connection,
                df=df_prepared,
                table_name=table_name,
                schema=self.conn.schema,
                quote_identifiers=False
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            result = {
                'status': 'success' if success else 'failed',
                'rows_loaded': num_rows,
                'chunks': num_chunks,
                'duration_seconds': elapsed
            }

            logger.info(f"  Loaded {table_name}: {num_rows} rows in {elapsed:.2f}s")

            # Log the load
            self._log_load(table_name, num_rows, 'success', None, elapsed)

            return result

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            logger.error(f"  Failed to load {table_name}: {error_msg}")

            # Log the failure
            self._log_load(table_name, 0, 'failed', error_msg, elapsed)

            return {
                'status': 'failed',
                'error': error_msg,
                'duration_seconds': elapsed
            }

    def _prepare_dataframe(self, df: pd.DataFrame, table_def: Dict) -> pd.DataFrame:
        """
        Prepare DataFrame for Snowflake loading.

        Args:
            df: Source DataFrame
            table_def: Table definition

        Returns:
            Prepared DataFrame
        """
        df_prepared = df.copy()

        # Get expected columns from table definition
        expected_columns = list(table_def['columns'].keys())

        # Remove auto-increment columns
        expected_columns = [
            col for col in expected_columns
            if 'AUTOINCREMENT' not in table_def['columns'].get(col, '')
        ]

        # Rename columns to uppercase
        df_prepared.columns = [col.upper() for col in df_prepared.columns]

        # Select only columns that exist in both DataFrame and table definition
        available_columns = [col for col in expected_columns if col in df_prepared.columns]
        df_prepared = df_prepared[available_columns]

        # Convert datetime columns
        for col in df_prepared.columns:
            if df_prepared[col].dtype == 'datetime64[ns]':
                df_prepared[col] = df_prepared[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif 'date' in col.lower() and df_prepared[col].dtype == 'object':
                try:
                    df_prepared[col] = pd.to_datetime(df_prepared[col]).dt.strftime('%Y-%m-%d')
                except:
                    pass

        # Handle NaN values
        df_prepared = df_prepared.fillna('')

        return df_prepared

    def _log_load(
        self,
        table_name: str,
        rows_loaded: int,
        status: str,
        error_message: str,
        duration: float
    ) -> None:
        """Log load operation to ETL_LOAD_LOG table."""
        self.load_log.append({
            'table_name': table_name,
            'rows_loaded': rows_loaded,
            'status': status,
            'error_message': error_message,
            'duration_seconds': duration,
            'timestamp': datetime.now()
        })

        try:
            sql = """
INSERT INTO ETL_LOAD_LOG (LOAD_TIMESTAMP, TABLE_NAME, ROWS_LOADED, STATUS, ERROR_MESSAGE, DURATION_SECONDS)
VALUES (CURRENT_TIMESTAMP(), %s, %s, %s, %s, %s)
"""
            self.conn.execute(sql, (table_name, rows_loaded, status, error_message, duration))
        except Exception as e:
            logger.warning(f"Could not log to ETL_LOAD_LOG: {e}")

    def _load_data_dictionary(self) -> None:
        """Load data dictionary metadata into Snowflake."""
        try:
            # Truncate existing
            self.conn.execute("TRUNCATE TABLE IF EXISTS DATA_DICTIONARY")

            # Insert data dictionary entries
            for dataset_key, dataset_info in DATA_DICTIONARY.items():
                source = dataset_info.get('source', '')

                for table_key, table_info in dataset_info.get('tables', {}).items():
                    table_name = f"{dataset_key}_{table_key}".upper()
                    description = table_info.get('description', '')

                    for column in table_info.get('columns', []):
                        sql = """
INSERT INTO DATA_DICTIONARY (TABLE_NAME, COLUMN_NAME, DATA_TYPE, DESCRIPTION, SOURCE, LAST_UPDATED)
VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
"""
                        self.conn.execute(sql, (table_name, column.upper(), 'VARCHAR', description, source))

            logger.info("  Loaded DATA_DICTIONARY metadata")

        except Exception as e:
            logger.warning(f"Could not load data dictionary: {e}")


# =============================================================================
# DATABASE MANAGER
# =============================================================================
class DatabaseManager:
    """
    High-level database management for Texas Governor race data.
    """

    def __init__(
        self,
        account: str = None,
        user: str = None,
        password: str = None,
        warehouse: str = None,
        database: str = None,
        schema: str = None
    ):
        """
        Initialize Database Manager.

        Args:
            account: Snowflake account
            user: Snowflake user
            password: Snowflake password
            warehouse: Snowflake warehouse
            database: Snowflake database
            schema: Snowflake schema
        """
        self.connection = SnowflakeConnection(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        self.schema_manager = None
        self.data_loader = None

    def connect(self) -> bool:
        """Connect to Snowflake."""
        if self.connection.connect():
            self.schema_manager = SchemaManager(self.connection)
            self.data_loader = DataLoader(self.connection)
            return True
        return False

    def disconnect(self) -> None:
        """Disconnect from Snowflake."""
        self.connection.disconnect()

    def setup_database(self, drop_existing: bool = False) -> Dict[str, bool]:
        """
        Set up database schema and tables.

        Args:
            drop_existing: If True, drop and recreate tables

        Returns:
            Dictionary of setup results
        """
        logger.info("=" * 60)
        logger.info("SETTING UP SNOWFLAKE DATABASE")
        logger.info("=" * 60)

        results = {}

        # Create schema
        results['schema'] = self.schema_manager.create_schema()

        # Create tables
        logger.info("\nCreating tables...")
        table_results = self.schema_manager.create_all_tables(drop_existing)
        results['tables'] = table_results

        # Summary
        success_count = sum(1 for v in table_results.values() if v)
        logger.info(f"\nSetup complete: {success_count}/{len(table_results)} tables created")

        return results

    def load_from_etl(
        self,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        truncate_first: bool = True
    ) -> Dict[str, Any]:
        """
        Run ETL pipeline and load data to Snowflake.

        Args:
            start_year: Start year for ETL
            end_year: End year for ETL
            truncate_first: If True, truncate tables before loading

        Returns:
            Dictionary of load results
        """
        logger.info("=" * 60)
        logger.info("LOADING DATA FROM ETL TO SNOWFLAKE")
        logger.info("=" * 60)

        # Run ETL pipeline
        logger.info("\nRunning ETL pipeline...")
        pipeline = ETLPipeline(
            start_year=start_year,
            end_year=end_year
        )
        etl_results = pipeline.run(
            extract=True,
            transform=True,
            load=False  # We'll load to Snowflake instead
        )

        # Get transformed data
        transformed_data = pipeline.transformed_data

        if not transformed_data:
            logger.error("No transformed data available from ETL")
            return {'status': 'failed', 'error': 'No ETL data'}

        # Load to Snowflake
        logger.info("\nLoading to Snowflake...")
        load_results = self.data_loader.load_all_data(
            transformed_data,
            truncate_first=truncate_first
        )

        # Summary
        success_count = sum(1 for v in load_results.values() if v.get('status') == 'success')
        total_rows = sum(v.get('rows_loaded', 0) for v in load_results.values())

        logger.info(f"\nLoad complete: {success_count}/{len(load_results)} tables loaded")
        logger.info(f"Total rows loaded: {total_rows:,}")

        return {
            'etl_results': etl_results,
            'load_results': load_results,
            'total_rows': total_rows
        }

    def run_query(self, sql: str) -> pd.DataFrame:
        """
        Run a SQL query and return results as DataFrame.

        Args:
            sql: SQL query to execute

        Returns:
            Query results as DataFrame
        """
        self.connection.execute(sql)
        columns = [desc[0] for desc in self.connection.cursor.description]
        data = self.connection.fetchall()
        return pd.DataFrame(data, columns=columns)

    def get_table_stats(self) -> pd.DataFrame:
        """
        Get statistics for all tables.

        Returns:
            DataFrame with table statistics
        """
        stats = []

        for table_name in SNOWFLAKE_TABLES.keys():
            if self.schema_manager.table_exists(table_name):
                row_count = self.schema_manager.get_table_row_count(table_name)
                stats.append({
                    'table_name': table_name,
                    'row_count': row_count,
                    'exists': True
                })
            else:
                stats.append({
                    'table_name': table_name,
                    'row_count': 0,
                    'exists': False
                })

        return pd.DataFrame(stats)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def print_snowflake_schema() -> None:
    """Print the Snowflake schema definition."""
    print("\n" + "=" * 70)
    print("SNOWFLAKE SCHEMA: TEXAS_GOVERNOR")
    print("=" * 70)

    for table_name, table_def in SNOWFLAKE_TABLES.items():
        print(f"\n{table_name}")
        print("-" * 50)
        print(f"  Description: {table_def.get('description', 'N/A')}")
        print(f"  Source Key: {table_def.get('source_key', 'N/A')}")
        print(f"  Primary Key: {', '.join(table_def.get('primary_key', []))}")
        print(f"  Columns:")

        for col_name, col_type in list(table_def['columns'].items())[:5]:
            print(f"    {col_name}: {col_type}")

        if len(table_def['columns']) > 5:
            print(f"    ... and {len(table_def['columns']) - 5} more columns")

    print("\n" + "=" * 70)


def generate_ddl() -> str:
    """
    Generate complete DDL for all tables.

    Returns:
        DDL string
    """
    ddl_lines = [
        "-- Snowflake DDL for Texas Governor Race Data",
        f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"CREATE SCHEMA IF NOT EXISTS {DEFAULT_SCHEMA};",
        f"USE SCHEMA {DEFAULT_SCHEMA};",
        ""
    ]

    for table_name, table_def in SNOWFLAKE_TABLES.items():
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
# MAIN
# =============================================================================
def main():
    """Main entry point for database operations."""
    parser = argparse.ArgumentParser(
        description='Snowflake Database Manager for Texas Governor Race Data'
    )

    parser.add_argument(
        '--setup', '-s',
        action='store_true',
        help='Create schema and tables in Snowflake'
    )
    parser.add_argument(
        '--load', '-l',
        action='store_true',
        help='Load data from ETL pipeline to Snowflake'
    )
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run full setup and load'
    )
    parser.add_argument(
        '--drop-existing',
        action='store_true',
        help='Drop existing tables before setup'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Run ad-hoc SQL query'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show table statistics'
    )
    parser.add_argument(
        '--print-schema',
        action='store_true',
        help='Print Snowflake schema definition'
    )
    parser.add_argument(
        '--generate-ddl',
        action='store_true',
        help='Generate DDL and save to file'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=DEFAULT_START_YEAR,
        help=f'Start year for data (default: {DEFAULT_START_YEAR})'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=DEFAULT_END_YEAR,
        help=f'End year for data (default: {DEFAULT_END_YEAR})'
    )

    args = parser.parse_args()

    # Handle non-connection operations
    if args.print_schema:
        print_snowflake_schema()
        return

    if args.generate_ddl:
        ddl = generate_ddl()
        ddl_file = 'snowflake_ddl.sql'
        with open(ddl_file, 'w') as f:
            f.write(ddl)
        print(f"DDL saved to {ddl_file}")
        print(ddl)
        return

    # Check if any action specified
    if not any([args.setup, args.load, args.full, args.query, args.stats]):
        parser.print_help()
        print("\nExamples:")
        print("  python database.py --setup              # Create tables")
        print("  python database.py --load               # Load data from ETL")
        print("  python database.py --full               # Setup + Load")
        print("  python database.py --stats              # Show table stats")
        print("  python database.py --print-schema       # Print schema")
        print("  python database.py --generate-ddl       # Generate DDL file")
        print("  python database.py --query 'SELECT...'  # Run query")
        return

    # Initialize database manager and connect
    db = DatabaseManager()

    if not db.connect():
        logger.error("Failed to connect to Snowflake. Check your credentials.")
        sys.exit(1)

    try:
        if args.full:
            args.setup = True
            args.load = True

        if args.setup:
            db.setup_database(drop_existing=args.drop_existing)

        if args.load:
            db.load_from_etl(
                start_year=args.start_year,
                end_year=args.end_year
            )

        if args.query:
            result = db.run_query(args.query)
            print(result.to_string())

        if args.stats:
            stats = db.get_table_stats()
            print("\nTable Statistics:")
            print(stats.to_string())

    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
