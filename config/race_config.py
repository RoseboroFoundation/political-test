"""
Race Configuration Module - Generalizable Election Framework

This module defines the configuration system for multi-race, multi-state election prediction.
Replaces hardcoded Texas Governor data with a flexible configuration system.

REFACTORING NOTES:
- [ADD] New file - centralizes all race-specific configuration
- [ADD] Supports Governor, US Senate, US House race types
- [ADD] State-agnostic design with state-specific data loading
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


class RaceType(Enum):
    """Enumeration of supported race types."""
    GOVERNOR = "governor"
    US_SENATE = "us_senate"
    US_HOUSE = "us_house"
    STATE_SENATE = "state_senate"
    STATE_HOUSE = "state_house"
    PRESIDENT = "president"


class ElectionContext(Enum):
    """Election year context."""
    PRESIDENTIAL = "presidential"  # Presidential election year
    MIDTERM = "midterm"           # Midterm election year
    OFF_YEAR = "off_year"         # Odd-year elections (some states)


@dataclass
class RaceConfig:
    """
    Configuration for a specific race type.

    Attributes:
        race_type: Type of race (Governor, Senate, etc.)
        state: Two-letter state code (None for federal aggregate)
        district: Congressional district (for US House only)
        election_years: List of election years with data
        cycle_length: Years between elections (4 for Governor, 6 for Senate, 2 for House)
        data_sources: Dictionary of data source configurations
        feature_availability: Which feature tiers are available for this race
    """
    race_type: RaceType
    state: str
    district: Optional[int] = None
    election_years: List[int] = field(default_factory=list)
    cycle_length: int = 4
    data_sources: Dict[str, str] = field(default_factory=dict)
    feature_availability: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Set defaults based on race type."""
        if self.race_type == RaceType.US_SENATE:
            self.cycle_length = 6
        elif self.race_type == RaceType.US_HOUSE:
            self.cycle_length = 2

        # Default feature availability (Tier 1 always available)
        if not self.feature_availability:
            self.feature_availability = {
                'tier1_universal': True,
                'tier2_polling': False,
                'tier2_sentiment': False,
                'tier2_vix': True,
                'tier3_coattails': False,
                'tier3_culture_war': False
            }

    @property
    def race_id(self) -> str:
        """Generate unique race identifier."""
        if self.district:
            return f"{self.state}_{self.race_type.value}_{self.district}"
        return f"{self.state}_{self.race_type.value}"

    @property
    def election_context(self) -> Dict[int, ElectionContext]:
        """Determine election context for each year."""
        contexts = {}
        for year in self.election_years:
            if year % 4 == 0:
                contexts[year] = ElectionContext.PRESIDENTIAL
            elif year % 2 == 0:
                contexts[year] = ElectionContext.MIDTERM
            else:
                contexts[year] = ElectionContext.OFF_YEAR
        return contexts


@dataclass
class CandidateInfo:
    """Information about a candidate in a specific race."""
    name: str
    party: str
    incumbent: bool = False
    filer_id: Optional[str] = None
    committee_name: Optional[str] = None


@dataclass
class ElectionResult:
    """Result of a single election."""
    year: int
    state: str
    race_type: RaceType
    district: Optional[int]
    winner: str
    winner_party: str
    margin_pct: float
    total_votes: int
    turnout_pct: float
    candidates: Dict[str, Dict]  # party -> {name, votes, pct, incumbent}
    election_date: str


# =============================================================================
# FEATURE TIER DEFINITIONS
# =============================================================================

FEATURE_TIERS = {
    'tier1_universal': {
        'description': 'Always available for any race',
        'features': [
            'partisan_lean',           # Historical partisan lean of state/district
            'incumbency',              # Binary: incumbent running
            'incumbent_party',         # Which party holds seat
            'open_seat',               # Binary: no incumbent
            'funding_ratio',           # R raised / D raised
            'total_funding',           # Total $ raised both candidates
            'national_environment',    # Generic ballot average
            'presidential_approval',   # Current president approval rating
            'gdp_growth',              # Latest GDP growth
            'unemployment_rate',       # Current unemployment
            'inflation_rate',          # Current CPI inflation
            'consumer_sentiment',      # Michigan consumer sentiment
            'election_context',        # Presidential/Midterm/Off-year
            'same_party_as_president', # Incumbent party same as president
        ]
    },
    'tier2_enhanced': {
        'description': 'Available when polling/market data exists',
        'features': [
            'poll_margin_mean',        # Average polling margin
            'poll_margin_std',         # Polling volatility
            'poll_count',              # Number of polls
            'poll_recency_days',       # Days since last poll
            'historical_polling_error', # State/race polling bias
            'vix_mean',                # VIX average pre-election
            'vix_trend',               # VIX direction (rising/falling)
            'news_sentiment_positive', # FinBERT positive sentiment
            'news_sentiment_negative', # FinBERT negative sentiment
            'news_coverage_ratio',     # R coverage / D coverage
        ]
    },
    'tier3_race_specific': {
        'description': 'Race-specific adjustments',
        'features': [
            'midterm_penalty',         # In-party penalty in midterms
            'coattail_effect',         # Presidential coattails
            'wave_year',               # Wave election indicator
            'culture_war_exposure',    # Culture war event count
            'candidate_quality',       # Quality score (experience, fundraising, scandals)
            'primary_divisiveness',    # Contested primary indicator
            'redistricting_effect',    # New district lines (House only)
        ]
    }
}


# =============================================================================
# STATE PARTISAN LEAN (Cook PVI style)
# =============================================================================

# Based on 2020 presidential + 2016 presidential average vs national
STATE_PARTISAN_LEAN = {
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
# CONFIGURATION LOADER
# =============================================================================

class RaceConfigLoader:
    """
    Loads and manages race configurations from JSON files.

    Expected directory structure:
    config/
        races/
            tx_governor.json
            ca_governor.json
            pa_us_senate.json
            historical_races.json  # Aggregated historical data
    """

    def __init__(self, config_dir: str = './config/races'):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, RaceConfig] = {}
        self.results: Dict[str, List[ElectionResult]] = {}

    def load_race_config(self, race_id: str) -> Optional[RaceConfig]:
        """Load configuration for a specific race."""
        config_file = self.config_dir / f"{race_id.lower()}.json"

        if not config_file.exists():
            return None

        with open(config_file, 'r') as f:
            data = json.load(f)

        config = RaceConfig(
            race_type=RaceType(data['race_type']),
            state=data['state'],
            district=data.get('district'),
            election_years=data['election_years'],
            cycle_length=data.get('cycle_length', 4),
            data_sources=data.get('data_sources', {}),
            feature_availability=data.get('feature_availability', {})
        )

        self.configs[race_id] = config
        return config

    def load_historical_results(self) -> Dict[str, List[ElectionResult]]:
        """
        Load aggregated historical results for training.

        This is the key data source for the hierarchical model.
        Expected to contain 500+ race results from 2010-2024.
        """
        results_file = self.config_dir / 'historical_races.json'

        if not results_file.exists():
            return {}

        with open(results_file, 'r') as f:
            data = json.load(f)

        for race_id, results in data.items():
            self.results[race_id] = [
                ElectionResult(
                    year=r['year'],
                    state=r['state'],
                    race_type=RaceType(r['race_type']),
                    district=r.get('district'),
                    winner=r['winner'],
                    winner_party=r['winner_party'],
                    margin_pct=r['margin_pct'],
                    total_votes=r['total_votes'],
                    turnout_pct=r['turnout_pct'],
                    candidates=r['candidates'],
                    election_date=r['election_date']
                )
                for r in results
            ]

        return self.results

    def get_all_races(self,
                      race_type: Optional[RaceType] = None,
                      min_elections: int = 3) -> List[RaceConfig]:
        """Get all race configs, optionally filtered by type."""
        configs = []

        for config_file in self.config_dir.glob('*.json'):
            if config_file.name == 'historical_races.json':
                continue

            config = self.load_race_config(config_file.stem)
            if config is None:
                continue

            if race_type and config.race_type != race_type:
                continue

            if len(config.election_years) < min_elections:
                continue

            configs.append(config)

        return configs


# =============================================================================
# EXAMPLE CONFIGURATION - Texas Governor (for migration)
# =============================================================================

def create_texas_governor_config() -> RaceConfig:
    """
    Create Texas Governor configuration.

    This replaces the hardcoded data in clean.py lines 5077-5173.
    """
    return RaceConfig(
        race_type=RaceType.GOVERNOR,
        state='TX',
        election_years=[2010, 2014, 2018, 2022, 2026],
        cycle_length=4,
        data_sources={
            'election_results': 'https://elections.sos.state.tx.us/',
            'campaign_finance': 'https://www.ethics.state.tx.us/',
            'polling': 'fivethirtyeight,rcp',
        },
        feature_availability={
            'tier1_universal': True,
            'tier2_polling': True,
            'tier2_sentiment': True,
            'tier2_vix': True,
            'tier3_coattails': True,
            'tier3_culture_war': True
        }
    )


def save_config_template():
    """Save a template configuration file."""
    config = create_texas_governor_config()

    template = {
        'race_type': config.race_type.value,
        'state': config.state,
        'district': config.district,
        'election_years': config.election_years,
        'cycle_length': config.cycle_length,
        'data_sources': config.data_sources,
        'feature_availability': config.feature_availability,
        'results': [
            {
                'year': 2022,
                'winner': 'Greg Abbott',
                'winner_party': 'R',
                'margin_pct': 11.1,
                'total_votes': 8076816,
                'turnout_pct': 45.6,
                'candidates': {
                    'R': {'name': 'Greg Abbott', 'votes': 4427320, 'pct': 54.8, 'incumbent': True},
                    'D': {'name': "Beto O'Rourke", 'votes': 3528219, 'pct': 43.7, 'incumbent': False}
                },
                'election_date': '2022-11-08'
            }
        ]
    }

    return template
