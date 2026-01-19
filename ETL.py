"""
ETL Pipeline for Texas Governor Race Political Data.

This module provides Extract, Transform, Load (ETL) functionality for
aggregating and processing Texas Governor race data from multiple sources:
- Election results from Texas Secretary of State
- Campaign finance data from Texas Ethics Commission
- Polling data from RealClearPolitics and Texas Politics Project
- News coverage from Guardian, NYT, and Texas media

Usage:
    python ETL.py --extract --transform --load
    python ETL.py --full-pipeline
    python ETL.py --extract-only
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
from typing import Dict, List, Optional, Any
import json

import pandas as pd

# Import from clean.py
from clean import (
    load_texas_governor_election_data,
    load_texas_campaign_finance_data,
    load_texas_governor_polling_data,
    load_texas_governor_news_data,
    get_texas_governor_election_summary,
    get_texas_campaign_finance_summary,
    get_texas_governor_polling_summary,
    get_texas_governor_news_summary
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

# Default paths
DEFAULT_DATA_DIR = './data'
DEFAULT_OUTPUT_DIR = './output'
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2025

# =============================================================================
# DATA DICTIONARY
# =============================================================================
"""
Complete Data Dictionary for Texas Governor Race Political Data

This ETL pipeline processes data from clean.py which contains the following datasets:

CULTURE WAR COMPANIES DATA (from clean.py):
-------------------------------------------
- culturewardata: Culture war companies events
- stockdata: Historical stock prices
- vixdata: VIX volatility index
- ff_factors: Fama-French factors (FF3, FF5, MOM)
- form4data: SEC Form 4 insider trading
- newsdata: News articles from Guardian, NYT, Reddit

MACROECONOMIC DATA (from clean.py):
-----------------------------------
- inflationdata: Inflation measures from FRED
- inflation_expectations: Breakeven inflation & survey expectations
- comprehensive_inflation: All inflation measures combined
- treasury_yields: Treasury yield curve (1M to 30Y, TIPS)
- policy_rates: Fed Funds, SOFR, Prime, discount rates
- credit_spreads: Corporate yields, credit spreads, mortgages
- comprehensive_rates: All rates with yield curve metrics
- industrial_production: IP indices, sectors, capacity utilization
- ip_growth: IP growth rates (YoY, MoM) and diffusion indices
- comprehensive_ip: All IP measures combined
- money_supply: M1, M2, monetary base, components
- money_velocity: M1 and M2 velocity
- fed_balance_sheet: Fed assets, reserves, balance sheet
- comprehensive_m2: All M2 measures with growth rates
- gdp_data: Nominal/Real GDP, growth rates, per capita
- gdp_components: Consumption, Investment, Government, Trade
- gdp_industry: GDP by industry/sector (value added)
- comprehensive_gdp: All GDP measures combined
- employment_data: Payrolls, unemployment rates, labor force
- jobless_claims: Initial/continuing claims, insured unemployment
- wages_hours: Average earnings, hours worked, labor costs
- jolts_data: Job openings, hires, quits, separations
- comprehensive_employment: All employment measures combined
- additional_macro: Consumer Sentiment, Housing, Dollar Index

TEXAS GOVERNOR RACE DATA (processed by this ETL):
-------------------------------------------------
- texas_governor_elections: Texas Governor election data (2010-2022)
    * statewide: Statewide results by candidate and party
        - election_year, election_date, race, state
        - candidate, party, votes, vote_percentage
        - total_votes, turnout_percentage, winner, incumbent
    * county: County-level results (when available)
        - election_year, election_date, county, state
        - race, candidate_index, votes
    * historical: Historical summary with margins and trends
        - election_year, winner, winner_party, winner_votes
        - runner_up, margin_percentage, margin_votes
        - turnout_percentage, incumbent_won, party_flip

- texas_campaign_finance: Texas Governor campaign finance data (2010-2022)
    * contributions: Individual contribution records
        - election_year, candidate, party, donor_name
        - amount, donor_employer, contribution_date, contribution_type
    * expenditures: Campaign expenditure records by category
        - election_year, candidate, party, committee
        - category, amount, percentage, expenditure_date
    * summary: Summary totals by candidate and cycle
        - election_year, candidate, party, incumbent, committee
        - total_raised, total_spent, cash_on_hand
        - individual_contributions, pac_contributions
        - num_contributors, avg_contribution, burn_rate
    * donors: Top donor analysis
        - donor_name, candidate, election_year
        - total_amount, num_contributions, avg_contribution

- texas_governor_polls: Texas Governor polling data (2010-2022)
    * polls: Individual poll records with results and methodology
        - pollster, start_date, end_date, mid_date
        - sample_size, population (LV/RV), moe
        - republican, democrat, other, margin
        - republican_candidate, democrat_candidate
        - election_year, days_to_election
    * averages: RCP-style polling averages by cycle
        - election_year, period (Full Cycle/Final Month)
        - republican_candidate, democrat_candidate, num_polls
        - avg_republican, avg_democrat, avg_margin
        - actual_republican, actual_democrat, actual_margin
        - polling_error
    * pollsters: Pollster ratings and methodology info
        - pollster, type, methodology
        - fivethirtyeight_rating, partisan_lean
        - transparency, typical_sample
    * trends: Polling trends over time within each cycle
        - election_year, initial_margin, final_margin
        - margin_change, trend_direction, volatility

- texas_governor_news: Texas Governor race news coverage (2010-2022)
    * articles: All news articles with metadata
        - candidate, party, election_year
        - source, title, url, published_date
        - snippet, author, section, word_count
        - scope (Texas/National), topic, sentiment
    * by_candidate: Article counts by candidate
        - election_year, candidate, party
        - total_articles, texas_articles, national_articles
        - texas_pct, top_topic
    * by_source: Article counts by source
        - source, election_year, article_count, candidates_covered
    * by_topic: Article counts by topic
        - topic, election_year, article_count, by_candidate
    * timeline: News volume over time
        - month, candidate, election_year, article_count
    * coverage_summary: Summary of coverage patterns
        - election_year, candidate, party
        - total_articles, texas_coverage, national_coverage
        - unique_sources, top_topic, coverage_ratio

ETL OUTPUT DATASETS:
--------------------
The ETL pipeline creates the following transformed datasets:

- election_results: Cleaned statewide election results
- election_historical: Historical election summaries
- finance_summary: Campaign finance summary by candidate
- finance_contributions: Individual contributions
- finance_expenditures: Expenditure records
- polls: Individual poll records
- poll_averages: Polling averages by cycle
- pollsters: Pollster information
- poll_trends: Polling trends
- news_articles: News article records
- news_coverage_summary: News coverage analysis
- news_by_topic: News by topic analysis
- candidate_master: Integrated candidate dataset
- election_cycle_summary: Summary by election cycle
- timeline: Unified timeline of events
"""

DATA_DICTIONARY = {
    'texas_governor_elections': {
        'description': 'Texas Governor election data (2010-2022)',
        'source': 'Texas Secretary of State',
        'url': 'https://www.sos.state.tx.us/elections/historical/',
        'tables': {
            'statewide': {
                'description': 'Statewide results by candidate and party',
                'columns': [
                    'election_year', 'election_date', 'race', 'state',
                    'candidate', 'party', 'votes', 'vote_percentage',
                    'total_votes', 'turnout_percentage', 'winner', 'incumbent'
                ]
            },
            'county': {
                'description': 'County-level results',
                'columns': [
                    'election_year', 'election_date', 'county', 'state',
                    'race', 'candidate_index', 'votes'
                ]
            },
            'historical': {
                'description': 'Historical summary with margins and trends',
                'columns': [
                    'election_year', 'winner', 'winner_party', 'winner_votes',
                    'runner_up', 'margin_percentage', 'margin_votes',
                    'turnout_percentage', 'incumbent_won', 'party_flip'
                ]
            }
        }
    },
    'texas_campaign_finance': {
        'description': 'Texas Governor campaign finance data (2010-2022)',
        'source': 'Texas Ethics Commission',
        'url': 'https://www.ethics.state.tx.us/search/cf/',
        'tables': {
            'contributions': {
                'description': 'Individual contribution records',
                'columns': [
                    'election_year', 'candidate', 'party', 'donor_name',
                    'amount', 'donor_employer', 'contribution_date', 'contribution_type'
                ]
            },
            'expenditures': {
                'description': 'Campaign expenditure records by category',
                'columns': [
                    'election_year', 'candidate', 'party', 'committee',
                    'category', 'amount', 'percentage', 'expenditure_date'
                ]
            },
            'summary': {
                'description': 'Summary totals by candidate and cycle',
                'columns': [
                    'election_year', 'candidate', 'party', 'incumbent', 'committee',
                    'total_raised', 'total_spent', 'cash_on_hand',
                    'individual_contributions', 'pac_contributions',
                    'num_contributors', 'avg_contribution', 'burn_rate'
                ]
            },
            'donors': {
                'description': 'Top donor analysis',
                'columns': [
                    'donor_name', 'candidate', 'election_year',
                    'total_amount', 'num_contributions', 'avg_contribution'
                ]
            }
        }
    },
    'texas_governor_polls': {
        'description': 'Texas Governor polling data (2010-2022)',
        'source': 'RealClearPolitics, Texas Politics Project',
        'url': 'https://www.realclearpolling.com/',
        'tables': {
            'polls': {
                'description': 'Individual poll records with results',
                'columns': [
                    'pollster', 'start_date', 'end_date', 'mid_date',
                    'sample_size', 'population', 'moe',
                    'republican', 'democrat', 'other', 'margin',
                    'republican_candidate', 'democrat_candidate',
                    'election_year', 'days_to_election'
                ]
            },
            'averages': {
                'description': 'Polling averages by cycle',
                'columns': [
                    'election_year', 'period', 'republican_candidate', 'democrat_candidate',
                    'num_polls', 'avg_republican', 'avg_democrat', 'avg_margin',
                    'actual_republican', 'actual_democrat', 'actual_margin', 'polling_error'
                ]
            },
            'pollsters': {
                'description': 'Pollster ratings and methodology',
                'columns': [
                    'pollster', 'type', 'methodology',
                    'fivethirtyeight_rating', 'partisan_lean', 'transparency', 'typical_sample'
                ]
            },
            'trends': {
                'description': 'Polling trends over time',
                'columns': [
                    'election_year', 'initial_margin', 'final_margin',
                    'margin_change', 'trend_direction', 'volatility'
                ]
            }
        }
    },
    'texas_governor_news': {
        'description': 'Texas Governor race news coverage (2010-2022)',
        'source': 'Guardian API, NYT API, Texas Tribune',
        'url': 'https://open-platform.theguardian.com/',
        'tables': {
            'articles': {
                'description': 'All news articles with metadata',
                'columns': [
                    'candidate', 'party', 'election_year',
                    'source', 'title', 'url', 'published_date',
                    'snippet', 'author', 'section', 'word_count',
                    'scope', 'topic', 'sentiment'
                ]
            },
            'by_candidate': {
                'description': 'Article counts by candidate',
                'columns': [
                    'election_year', 'candidate', 'party',
                    'total_articles', 'texas_articles', 'national_articles',
                    'texas_pct', 'top_topic'
                ]
            },
            'by_source': {
                'description': 'Article counts by source',
                'columns': ['source', 'election_year', 'article_count', 'candidates_covered']
            },
            'by_topic': {
                'description': 'Article counts by topic',
                'columns': ['topic', 'election_year', 'article_count', 'by_candidate']
            },
            'timeline': {
                'description': 'News volume over time',
                'columns': ['month', 'candidate', 'election_year', 'article_count']
            },
            'coverage_summary': {
                'description': 'Summary of coverage patterns',
                'columns': [
                    'election_year', 'candidate', 'party',
                    'total_articles', 'texas_coverage', 'national_coverage',
                    'unique_sources', 'top_topic', 'coverage_ratio'
                ]
            }
        }
    },
    'culture_war_companies': {
        'description': 'Culture war companies events and market impact data',
        'source': 'Roseboro Foundation Research',
        'url': 'internal',
        'tables': {
            'events': {
                'description': 'Culture war events by company',
                'columns': [
                    'company', 'year', 'culture_war_event', 'event_date',
                    'industry', 'ticker', 'estimated_political_leaning',
                    'political_leaning_justifications', 'naics_code',
                    'control_firm', 'control_ticker', 'rationale'
                ]
            },
            'stock_impact': {
                'description': 'Stock price impact around culture war events',
                'columns': [
                    'ticker', 'event_date', 'price_before', 'price_after',
                    'return_1d', 'return_5d', 'return_30d', 'abnormal_return',
                    'volume_change', 'volatility'
                ]
            },
            'insider_trading': {
                'description': 'Form 4 insider trading around events',
                'columns': [
                    'ticker', 'filing_date', 'transaction_date', 'insider_name',
                    'insider_title', 'transaction_type', 'shares', 'price',
                    'value', 'shares_owned_after'
                ]
            },
            'news_coverage': {
                'description': 'News coverage of culture war events',
                'columns': [
                    'company', 'ticker', 'event_date', 'source', 'title',
                    'published_date', 'sentiment', 'article_url'
                ]
            },
            'summary': {
                'description': 'Summary statistics by company and event',
                'columns': [
                    'company', 'ticker', 'industry', 'political_leaning',
                    'event_count', 'avg_stock_impact', 'avg_abnormal_return',
                    'news_article_count', 'insider_trades_count'
                ]
            }
        }
    }
}


def get_data_dictionary() -> Dict[str, Any]:
    """
    Return the complete data dictionary.

    Returns:
        Dictionary containing metadata for all datasets
    """
    return DATA_DICTIONARY


def print_data_dictionary() -> None:
    """Print formatted data dictionary to console."""
    print("\n" + "=" * 70)
    print("TEXAS GOVERNOR RACE - DATA DICTIONARY")
    print("=" * 70)

    for dataset_key, dataset_info in DATA_DICTIONARY.items():
        print(f"\n{dataset_key}")
        print("-" * 50)
        print(f"  Description: {dataset_info['description']}")
        print(f"  Source: {dataset_info['source']}")
        print(f"  URL: {dataset_info['url']}")
        print(f"  Tables:")

        for table_name, table_info in dataset_info['tables'].items():
            print(f"\n    {table_name}:")
            print(f"      {table_info['description']}")
            print(f"      Columns: {', '.join(table_info['columns'][:5])}...")

    print("\n" + "=" * 70)


# Data source configurations
DATA_SOURCES = {
    'elections': {
        'name': 'Texas Governor Elections',
        'loader': load_texas_governor_election_data,
        'cache_path': './data/elections',
        'summary_func': get_texas_governor_election_summary
    },
    'campaign_finance': {
        'name': 'Texas Campaign Finance',
        'loader': load_texas_campaign_finance_data,
        'cache_path': './data/campaign_finance',
        'summary_func': get_texas_campaign_finance_summary
    },
    'polling': {
        'name': 'Texas Governor Polling',
        'loader': load_texas_governor_polling_data,
        'cache_path': './data/polling',
        'summary_func': get_texas_governor_polling_summary
    },
    'news': {
        'name': 'Texas Governor News',
        'loader': load_texas_governor_news_data,
        'cache_path': './data/news',
        'summary_func': get_texas_governor_news_summary
    }
}


# =============================================================================
# EXTRACT
# =============================================================================
class Extractor:
    """
    Extracts data from various sources for Texas Governor race analysis.
    """

    def __init__(
        self,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        data_dir: str = DEFAULT_DATA_DIR
    ):
        """
        Initialize the Extractor.

        Args:
            start_year: Start year for data extraction
            end_year: End year for data extraction
            data_dir: Base directory for data storage
        """
        self.start_year = start_year
        self.end_year = end_year
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.extracted_data: Dict[str, Any] = {}
        self.extraction_metadata: Dict[str, Any] = {}

    def extract_all(self, sources: List[str] = None) -> Dict[str, Any]:
        """
        Extract data from all configured sources.

        Args:
            sources: List of source names to extract (default: all)

        Returns:
            Dictionary of extracted data by source
        """
        if sources is None:
            sources = list(DATA_SOURCES.keys())

        logger.info(f"Starting extraction for sources: {sources}")
        logger.info(f"Date range: {self.start_year} - {self.end_year}")

        for source_key in sources:
            if source_key not in DATA_SOURCES:
                logger.warning(f"Unknown source: {source_key}, skipping")
                continue

            self._extract_source(source_key)

        self._record_extraction_metadata()

        return self.extracted_data

    def _extract_source(self, source_key: str) -> None:
        """
        Extract data from a single source.

        Args:
            source_key: Key identifying the data source
        """
        source_config = DATA_SOURCES[source_key]
        logger.info(f"Extracting: {source_config['name']}...")

        try:
            start_time = datetime.now()

            # Call the loader function
            loader = source_config['loader']

            if source_key == 'news':
                data = loader(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cache_path=source_config['cache_path'],
                    refresh=False
                )
            else:
                data = loader(
                    start_year=self.start_year,
                    end_year=self.end_year,
                    cache_path=source_config['cache_path']
                )

            elapsed = (datetime.now() - start_time).total_seconds()

            self.extracted_data[source_key] = data
            self.extraction_metadata[source_key] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': elapsed,
                'record_counts': self._count_records(data)
            }

            logger.info(f"  Extracted {source_config['name']} in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"  Failed to extract {source_config['name']}: {e}")
            self.extracted_data[source_key] = None
            self.extraction_metadata[source_key] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _count_records(self, data: Any) -> Dict[str, int]:
        """Count records in extracted data."""
        counts = {}

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    counts[key] = len(value)
                elif isinstance(value, dict):
                    counts[key] = len(value)
                elif value is not None:
                    counts[key] = 1
                else:
                    counts[key] = 0
        elif isinstance(data, pd.DataFrame):
            counts['total'] = len(data)

        return counts

    def _record_extraction_metadata(self) -> None:
        """Record extraction metadata to file."""
        metadata_file = self.data_dir / 'extraction_metadata.json'

        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'start_year': self.start_year,
            'end_year': self.end_year,
            'sources': self.extraction_metadata
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Extraction metadata saved to {metadata_file}")


# =============================================================================
# TRANSFORM
# =============================================================================
class Transformer:
    """
    Transforms extracted data for analysis and loading.
    """

    def __init__(self, extracted_data: Dict[str, Any]):
        """
        Initialize the Transformer.

        Args:
            extracted_data: Dictionary of extracted data from Extractor
        """
        self.raw_data = extracted_data
        self.transformed_data: Dict[str, pd.DataFrame] = {}

    def transform_all(self) -> Dict[str, pd.DataFrame]:
        """
        Transform all extracted data.

        Returns:
            Dictionary of transformed DataFrames
        """
        logger.info("Starting data transformation...")

        # Transform each data source
        self._transform_elections()
        self._transform_campaign_finance()
        self._transform_polling()
        self._transform_news()

        # Create integrated datasets
        self._create_candidate_master()
        self._create_election_cycle_summary()
        self._create_timeline_dataset()

        logger.info(f"Transformation complete: {len(self.transformed_data)} datasets")

        return self.transformed_data

    def _transform_elections(self) -> None:
        """Transform election data."""
        elections = self.raw_data.get('elections')

        if elections is None:
            logger.warning("No election data to transform")
            return

        # Statewide results
        if 'statewide' in elections and elections['statewide'] is not None:
            df = elections['statewide'].copy()
            df['data_source'] = 'Texas SOS'
            df['last_updated'] = datetime.now().isoformat()
            self.transformed_data['election_results'] = df

        # Historical summary
        if 'historical' in elections and elections['historical'] is not None:
            df = elections['historical'].copy()
            self.transformed_data['election_historical'] = df

        logger.info("  Transformed election data")

    def _transform_campaign_finance(self) -> None:
        """Transform campaign finance data."""
        finance = self.raw_data.get('campaign_finance')

        if finance is None:
            logger.warning("No campaign finance data to transform")
            return

        # Summary data
        if 'summary' in finance and finance['summary'] is not None:
            df = finance['summary'].copy()
            df['data_source'] = 'Texas Ethics Commission'
            df['last_updated'] = datetime.now().isoformat()
            self.transformed_data['finance_summary'] = df

        # Contributions
        if 'contributions' in finance and finance['contributions'] is not None:
            df = finance['contributions'].copy()
            self.transformed_data['finance_contributions'] = df

        # Expenditures
        if 'expenditures' in finance and finance['expenditures'] is not None:
            df = finance['expenditures'].copy()
            self.transformed_data['finance_expenditures'] = df

        logger.info("  Transformed campaign finance data")

    def _transform_polling(self) -> None:
        """Transform polling data."""
        polling = self.raw_data.get('polling')

        if polling is None:
            logger.warning("No polling data to transform")
            return

        # Individual polls
        if 'polls' in polling and polling['polls'] is not None:
            df = polling['polls'].copy()
            df['data_source'] = 'RCP/TPP'
            df['last_updated'] = datetime.now().isoformat()
            self.transformed_data['polls'] = df

        # Polling averages
        if 'averages' in polling and polling['averages'] is not None:
            df = polling['averages'].copy()
            self.transformed_data['poll_averages'] = df

        # Pollster info
        if 'pollsters' in polling and polling['pollsters'] is not None:
            df = polling['pollsters'].copy()
            self.transformed_data['pollsters'] = df

        # Trends
        if 'trends' in polling and polling['trends'] is not None:
            df = polling['trends'].copy()
            self.transformed_data['poll_trends'] = df

        logger.info("  Transformed polling data")

    def _transform_news(self) -> None:
        """Transform news data."""
        news = self.raw_data.get('news')

        if news is None:
            logger.warning("No news data to transform")
            return

        # Articles
        if 'articles' in news and news['articles'] is not None:
            df = news['articles'].copy()
            df['last_updated'] = datetime.now().isoformat()
            self.transformed_data['news_articles'] = df

        # Coverage summary
        if 'coverage_summary' in news and news['coverage_summary'] is not None:
            df = news['coverage_summary'].copy()
            self.transformed_data['news_coverage_summary'] = df

        # By topic
        if 'by_topic' in news and news['by_topic'] is not None:
            df = news['by_topic'].copy()
            self.transformed_data['news_by_topic'] = df

        logger.info("  Transformed news data")

    def _create_candidate_master(self) -> None:
        """Create master candidate dataset combining all sources."""
        candidates = []

        # Get unique candidates from election results
        if 'election_results' in self.transformed_data:
            election_df = self.transformed_data['election_results']

            for _, row in election_df[election_df['winner'] == True].drop_duplicates(
                subset=['election_year', 'candidate']
            ).iterrows():
                candidates.append({
                    'candidate': row['candidate'],
                    'party': row['party'],
                    'election_year': row['election_year'],
                    'won': True
                })

            for _, row in election_df[election_df['winner'] == False].drop_duplicates(
                subset=['election_year', 'candidate']
            ).iterrows():
                # Only add major party candidates
                if row['party'] in ['R', 'D']:
                    candidates.append({
                        'candidate': row['candidate'],
                        'party': row['party'],
                        'election_year': row['election_year'],
                        'won': False
                    })

        if not candidates:
            return

        candidate_df = pd.DataFrame(candidates).drop_duplicates()

        # Merge with finance data
        if 'finance_summary' in self.transformed_data:
            finance_df = self.transformed_data['finance_summary'][
                ['candidate', 'election_year', 'total_raised', 'total_spent', 'num_contributors']
            ]
            candidate_df = candidate_df.merge(
                finance_df,
                on=['candidate', 'election_year'],
                how='left'
            )

        # Merge with polling data
        if 'poll_averages' in self.transformed_data:
            poll_df = self.transformed_data['poll_averages'][
                self.transformed_data['poll_averages']['period'] == 'Final Month'
            ][['election_year', 'avg_margin', 'polling_error']].drop_duplicates()

            candidate_df = candidate_df.merge(
                poll_df,
                on='election_year',
                how='left'
            )

        # Merge with news coverage
        if 'news_coverage_summary' in self.transformed_data:
            news_df = self.transformed_data['news_coverage_summary'][
                ['candidate', 'election_year', 'total_articles', 'top_topic']
            ]
            candidate_df = candidate_df.merge(
                news_df,
                on=['candidate', 'election_year'],
                how='left'
            )

        self.transformed_data['candidate_master'] = candidate_df
        logger.info("  Created candidate master dataset")

    def _create_election_cycle_summary(self) -> None:
        """Create summary dataset for each election cycle."""
        cycles = []

        election_years = [2010, 2014, 2018, 2022]

        for year in election_years:
            cycle = {'election_year': year}

            # Election results
            if 'election_historical' in self.transformed_data:
                hist = self.transformed_data['election_historical']
                year_hist = hist[hist['election_year'] == year]

                if not year_hist.empty:
                    row = year_hist.iloc[0]
                    cycle['winner'] = row['winner']
                    cycle['winner_party'] = row['winner_party']
                    cycle['margin_pct'] = row['margin_percentage']
                    cycle['total_votes'] = row['total_votes']
                    cycle['turnout_pct'] = row['turnout_percentage']

            # Finance totals
            if 'finance_summary' in self.transformed_data:
                fin = self.transformed_data['finance_summary']
                year_fin = fin[fin['election_year'] == year]

                if not year_fin.empty:
                    cycle['total_raised_all'] = year_fin['total_raised'].sum()
                    cycle['total_spent_all'] = year_fin['total_spent'].sum()
                    r_raised = year_fin[year_fin['party'] == 'R']['total_raised'].sum()
                    d_raised = year_fin[year_fin['party'] == 'D']['total_raised'].sum()
                    cycle['r_fundraising_advantage'] = r_raised - d_raised

            # Polling
            if 'poll_averages' in self.transformed_data:
                poll = self.transformed_data['poll_averages']
                year_poll = poll[
                    (poll['election_year'] == year) &
                    (poll['period'] == 'Full Cycle')
                ]

                if not year_poll.empty:
                    row = year_poll.iloc[0]
                    cycle['poll_avg_margin'] = row['avg_margin']
                    cycle['polling_error'] = row['polling_error']
                    cycle['num_polls'] = row['num_polls']

            # News
            if 'news_coverage_summary' in self.transformed_data:
                news = self.transformed_data['news_coverage_summary']
                year_news = news[news['election_year'] == year]

                if not year_news.empty:
                    cycle['total_news_articles'] = year_news['total_articles'].sum()

            cycles.append(cycle)

        self.transformed_data['election_cycle_summary'] = pd.DataFrame(cycles)
        logger.info("  Created election cycle summary")

    def _create_timeline_dataset(self) -> None:
        """Create unified timeline dataset for time series analysis."""
        timeline_records = []

        # Get polling timeline
        if 'polls' in self.transformed_data:
            polls = self.transformed_data['polls']

            for _, row in polls.iterrows():
                timeline_records.append({
                    'date': row.get('mid_date') or row.get('end_date'),
                    'election_year': row['election_year'],
                    'event_type': 'poll',
                    'source': row['pollster'],
                    'value': row['margin'],
                    'description': f"{row['pollster']}: R+{row['margin']}"
                })

        # Get news timeline
        if 'news_articles' in self.transformed_data:
            news = self.transformed_data['news_articles']

            for _, row in news.iterrows():
                timeline_records.append({
                    'date': row.get('published_date'),
                    'election_year': row['election_year'],
                    'event_type': 'news',
                    'source': row['source'],
                    'value': None,
                    'description': row['title'][:100] if row.get('title') else ''
                })

        if timeline_records:
            timeline_df = pd.DataFrame(timeline_records)
            timeline_df['date'] = pd.to_datetime(timeline_df['date'], utc=True)
            timeline_df = timeline_df.sort_values('date')
            self.transformed_data['timeline'] = timeline_df
            logger.info("  Created timeline dataset")


# =============================================================================
# LOAD
# =============================================================================
class Loader:
    """
    Loads transformed data to various destinations.
    """

    def __init__(
        self,
        transformed_data: Dict[str, pd.DataFrame],
        output_dir: str = DEFAULT_OUTPUT_DIR
    ):
        """
        Initialize the Loader.

        Args:
            transformed_data: Dictionary of transformed DataFrames
            output_dir: Output directory for loaded data
        """
        self.data = transformed_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_all(
        self,
        formats: List[str] = None,
        include_summary: bool = True
    ) -> Dict[str, str]:
        """
        Load all transformed data to specified formats.

        Args:
            formats: List of output formats ('csv', 'parquet', 'json', 'excel')
            include_summary: Whether to generate summary report

        Returns:
            Dictionary of output file paths
        """
        if formats is None:
            formats = ['csv']

        logger.info(f"Loading data to formats: {formats}")

        output_files = {}

        for format_type in formats:
            if format_type == 'csv':
                files = self._load_to_csv()
            elif format_type == 'parquet':
                files = self._load_to_parquet()
            elif format_type == 'json':
                files = self._load_to_json()
            elif format_type == 'excel':
                files = self._load_to_excel()
            else:
                logger.warning(f"Unknown format: {format_type}")
                continue

            output_files[format_type] = files

        if include_summary:
            summary_file = self._generate_summary_report()
            output_files['summary'] = summary_file

        self._save_load_manifest(output_files)

        return output_files

    def _load_to_csv(self) -> Dict[str, str]:
        """Load data to CSV files."""
        csv_dir = self.output_dir / 'csv'
        csv_dir.mkdir(exist_ok=True)

        files = {}

        for name, df in self.data.items():
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                filepath = csv_dir / f'{name}.csv'
                df.to_csv(filepath, index=False)
                files[name] = str(filepath)
                logger.info(f"  Saved {name}.csv ({len(df)} rows)")

        return files

    def _load_to_parquet(self) -> Dict[str, str]:
        """Load data to Parquet files."""
        parquet_dir = self.output_dir / 'parquet'
        parquet_dir.mkdir(exist_ok=True)

        files = {}

        for name, df in self.data.items():
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                filepath = parquet_dir / f'{name}.parquet'
                try:
                    df.to_parquet(filepath, index=False)
                    files[name] = str(filepath)
                    logger.info(f"  Saved {name}.parquet ({len(df)} rows)")
                except Exception as e:
                    logger.warning(f"  Could not save {name} to parquet: {e}")

        return files

    def _load_to_json(self) -> Dict[str, str]:
        """Load data to JSON files."""
        json_dir = self.output_dir / 'json'
        json_dir.mkdir(exist_ok=True)

        files = {}

        for name, df in self.data.items():
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                filepath = json_dir / f'{name}.json'
                df.to_json(filepath, orient='records', date_format='iso', indent=2)
                files[name] = str(filepath)
                logger.info(f"  Saved {name}.json ({len(df)} rows)")

        return files

    def _load_to_excel(self) -> Dict[str, str]:
        """Load data to Excel file with multiple sheets."""
        excel_file = self.output_dir / 'texas_governor_data.xlsx'

        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for name, df in self.data.items():
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        # Excel sheet names limited to 31 chars
                        sheet_name = name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"  Saved Excel workbook: {excel_file}")
            return {'workbook': str(excel_file)}

        except Exception as e:
            logger.warning(f"  Could not save Excel file: {e}")
            return {}

    def _generate_summary_report(self) -> str:
        """Generate a summary report of the loaded data."""
        report_file = self.output_dir / 'etl_summary_report.txt'

        lines = [
            "=" * 70,
            "TEXAS GOVERNOR RACE DATA - ETL SUMMARY REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            ""
        ]

        # Dataset summaries
        lines.append("DATASETS LOADED:")
        lines.append("-" * 50)

        for name, df in self.data.items():
            if df is not None and isinstance(df, pd.DataFrame):
                lines.append(f"  {name}:")
                lines.append(f"    Rows: {len(df):,}")
                lines.append(f"    Columns: {len(df.columns)}")
                lines.append(f"    Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                lines.append("")

        # Election cycle summary
        if 'election_cycle_summary' in self.data:
            lines.append("")
            lines.append("ELECTION CYCLE SUMMARY:")
            lines.append("-" * 50)

            cycle_df = self.data['election_cycle_summary']
            for _, row in cycle_df.iterrows():
                lines.append(f"  {int(row['election_year'])}:")
                if 'winner' in row:
                    lines.append(f"    Winner: {row.get('winner', 'N/A')} ({row.get('winner_party', 'N/A')})")
                if 'margin_pct' in row and pd.notna(row['margin_pct']):
                    lines.append(f"    Margin: {row['margin_pct']:.1f}%")
                if 'total_raised_all' in row and pd.notna(row['total_raised_all']):
                    lines.append(f"    Total Raised: ${row['total_raised_all']:,.0f}")
                if 'num_polls' in row and pd.notna(row['num_polls']):
                    lines.append(f"    Polls: {int(row['num_polls'])}")
                lines.append("")

        lines.append("=" * 70)

        with open(report_file, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"  Generated summary report: {report_file}")
        return str(report_file)

    def _save_load_manifest(self, output_files: Dict) -> None:
        """Save manifest of loaded files."""
        manifest_file = self.output_dir / 'load_manifest.json'

        manifest = {
            'load_timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'datasets_loaded': list(self.data.keys()),
            'output_files': output_files
        }

        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Load manifest saved to {manifest_file}")


# =============================================================================
# ETL PIPELINE
# =============================================================================
class ETLPipeline:
    """
    Complete ETL Pipeline for Texas Governor race data.
    """

    def __init__(
        self,
        start_year: int = DEFAULT_START_YEAR,
        end_year: int = DEFAULT_END_YEAR,
        data_dir: str = DEFAULT_DATA_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR
    ):
        """
        Initialize the ETL Pipeline.

        Args:
            start_year: Start year for data
            end_year: End year for data
            data_dir: Directory for raw/cached data
            output_dir: Directory for output data
        """
        self.start_year = start_year
        self.end_year = end_year
        self.data_dir = data_dir
        self.output_dir = output_dir

        self.extractor: Optional[Extractor] = None
        self.transformer: Optional[Transformer] = None
        self.loader: Optional[Loader] = None

        self.extracted_data: Dict = {}
        self.transformed_data: Dict = {}

    def run(
        self,
        extract: bool = True,
        transform: bool = True,
        load: bool = True,
        sources: List[str] = None,
        output_formats: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run the ETL pipeline.

        Args:
            extract: Whether to run extraction
            transform: Whether to run transformation
            load: Whether to run loading
            sources: Data sources to extract
            output_formats: Output formats for loading

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("STARTING ETL PIPELINE")
        logger.info(f"Date range: {self.start_year} - {self.end_year}")
        logger.info("=" * 60)

        results = {
            'start_time': datetime.now().isoformat(),
            'parameters': {
                'start_year': self.start_year,
                'end_year': self.end_year,
                'sources': sources or list(DATA_SOURCES.keys()),
                'output_formats': output_formats or ['csv']
            }
        }

        # EXTRACT
        if extract:
            logger.info("")
            logger.info("PHASE 1: EXTRACT")
            logger.info("-" * 40)

            self.extractor = Extractor(
                start_year=self.start_year,
                end_year=self.end_year,
                data_dir=self.data_dir
            )
            self.extracted_data = self.extractor.extract_all(sources)
            results['extraction'] = self.extractor.extraction_metadata

        # TRANSFORM
        if transform:
            logger.info("")
            logger.info("PHASE 2: TRANSFORM")
            logger.info("-" * 40)

            if not self.extracted_data:
                logger.error("No extracted data available for transformation")
            else:
                self.transformer = Transformer(self.extracted_data)
                self.transformed_data = self.transformer.transform_all()
                results['transformation'] = {
                    'datasets_created': list(self.transformed_data.keys()),
                    'total_datasets': len(self.transformed_data)
                }

        # LOAD
        if load:
            logger.info("")
            logger.info("PHASE 3: LOAD")
            logger.info("-" * 40)

            if not self.transformed_data:
                logger.error("No transformed data available for loading")
            else:
                self.loader = Loader(
                    transformed_data=self.transformed_data,
                    output_dir=self.output_dir
                )
                output_files = self.loader.load_all(formats=output_formats)
                results['loading'] = {
                    'output_directory': self.output_dir,
                    'output_files': output_files
                }

        results['end_time'] = datetime.now().isoformat()

        logger.info("")
        logger.info("=" * 60)
        logger.info("ETL PIPELINE COMPLETE")
        logger.info("=" * 60)

        return results

    def print_summaries(self) -> None:
        """Print summaries from all data sources."""
        if not self.extracted_data:
            logger.warning("No data extracted. Run extraction first.")
            return

        for source_key, config in DATA_SOURCES.items():
            if source_key in self.extracted_data and self.extracted_data[source_key]:
                try:
                    config['summary_func'](self.extracted_data[source_key])
                except Exception as e:
                    logger.warning(f"Could not print summary for {source_key}: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point for ETL pipeline."""
    parser = argparse.ArgumentParser(
        description='ETL Pipeline for Texas Governor Race Political Data'
    )

    parser.add_argument(
        '--extract', '-e',
        action='store_true',
        help='Run extraction phase'
    )
    parser.add_argument(
        '--transform', '-t',
        action='store_true',
        help='Run transformation phase'
    )
    parser.add_argument(
        '--load', '-l',
        action='store_true',
        help='Run loading phase'
    )
    parser.add_argument(
        '--full-pipeline', '-f',
        action='store_true',
        help='Run full ETL pipeline (extract, transform, load)'
    )
    parser.add_argument(
        '--sources', '-s',
        nargs='+',
        choices=['elections', 'campaign_finance', 'polling', 'news'],
        help='Data sources to process'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['csv', 'parquet', 'json', 'excel'],
        default=['csv'],
        help='Output formats (default: csv)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=DEFAULT_START_YEAR,
        help=f'Start year (default: {DEFAULT_START_YEAR})'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=DEFAULT_END_YEAR,
        help=f'End year (default: {DEFAULT_END_YEAR})'
    )
    parser.add_argument(
        '--data-dir',
        default=DEFAULT_DATA_DIR,
        help=f'Data directory (default: {DEFAULT_DATA_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--print-summaries',
        action='store_true',
        help='Print data summaries after extraction'
    )

    args = parser.parse_args()

    # Determine which phases to run
    if args.full_pipeline:
        extract = transform = load = True
    else:
        extract = args.extract
        transform = args.transform
        load = args.load

    # If no phases specified, show help
    if not any([extract, transform, load]):
        parser.print_help()
        print("\nExample usage:")
        print("  python ETL.py --full-pipeline")
        print("  python ETL.py --extract --transform --load")
        print("  python ETL.py -e -t -l --formats csv json")
        print("  python ETL.py --extract-only --sources elections polling")
        return

    # Create and run pipeline
    pipeline = ETLPipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    results = pipeline.run(
        extract=extract,
        transform=transform,
        load=load,
        sources=args.sources,
        output_formats=args.formats
    )

    # Print summaries if requested
    if args.print_summaries:
        pipeline.print_summaries()

    # Save results
    results_file = Path(args.output_dir) / 'etl_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nETL results saved to: {results_file}")


if __name__ == "__main__":
    main()
