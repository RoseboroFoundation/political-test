"""
Texas Governor Race Analysis - Executive Dashboard

A comprehensive Streamlit dashboard for analyzing Texas Governor race data
including elections, campaign finance, polling, news coverage, culture war
events, market data, and macroeconomic indicators.

Usage:
    streamlit run streamlit_app.py

Tabs:
    1. Client Race Results - Executive overview of election outcomes
    2. Model Test Results - Statistical analysis and model performance
    3. Academic - Raw data, code, and methodology for research
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Import project modules
from Model import StatisticalModelManager
from visualizations import Visualizer, COLORS
from ETL import ETLPipeline, DATA_DICTIONARY

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Texas Governor Race Analysis",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-title {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #1f4e79;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .tab-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .code-box {
        background-color: #263238;
        color: #aed581;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem 0;
        border-top: 1px solid #e9ecef;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_resource
def load_model_manager():
    """Load and cache the statistical model manager."""
    manager = StatisticalModelManager(use_database=False)
    manager.initialize()
    manager.run_all_statistics()
    return manager


@st.cache_data
def load_raw_data():
    """Load raw data files for academic tab."""
    raw_data = {}

    # Load CSV files from output directory
    csv_dir = './output/csv'
    if os.path.exists(csv_dir):
        for filename in os.listdir(csv_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(csv_dir, filename)
                try:
                    raw_data[filename.replace('.csv', '')] = pd.read_csv(filepath)
                except Exception as e:
                    st.warning(f"Could not load {filename}: {e}")

    # Load culture war data
    culture_war_path = './Culture_War_Companies_160_fullmeta.csv'
    if os.path.exists(culture_war_path):
        try:
            raw_data['culture_war_events'] = pd.read_csv(culture_war_path)
        except:
            pass

    # Load VIX data
    vix_path = './vix_data_2000_2025.csv'
    if os.path.exists(vix_path):
        try:
            raw_data['vix_data'] = pd.read_csv(vix_path)
        except:
            pass

    # Load macro data
    macro_path = './full_macro_data_2000_2025.csv'
    if os.path.exists(macro_path):
        try:
            raw_data['macro_data'] = pd.read_csv(macro_path)
        except:
            pass

    return raw_data


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def display_metric_card(col, title: str, value: Any, subtitle: str = None):
    """Display a styled metric card."""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div style="color: #6c757d; font-size: 0.8rem;">{subtitle}</div>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)


def format_currency(value):
    """Format value as currency."""
    if isinstance(value, (int, float)):
        return f"${value:,.0f}"
    return value


def format_percentage(value):
    """Format value as percentage."""
    if isinstance(value, (int, float)):
        return f"{value:.1f}%"
    return value


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render the sidebar with navigation and info."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/texas.png", width=80)
        st.title("Texas Governor Race")
        st.markdown("**Analysis Dashboard**")

        st.markdown("---")

        st.markdown("### About")
        st.markdown("""
        This dashboard provides comprehensive analysis of Texas Governor
        races from 2010-2024, including:

        - Election results & margins
        - Campaign finance data
        - Polling analysis
        - News coverage metrics
        - Culture war events
        - Market & economic context
        """)

        st.markdown("---")

        st.markdown("### Data Sources")
        st.markdown("""
        - Texas Secretary of State
        - FEC Campaign Finance
        - FiveThirtyEight Polls
        - Guardian/NYT News APIs
        - Yahoo Finance
        - FRED Economic Data
        """)

        st.markdown("---")

        st.markdown("### Quick Stats")
        try:
            manager = load_model_manager()
            elections = manager.all_results.get('elections', {})
            if elections:
                st.metric("Elections Analyzed", "4")
                st.metric("Years Covered", "2010-2022")
        except:
            st.info("Loading data...")

        st.markdown("---")
        st.markdown(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")


# =============================================================================
# TAB 1: CLIENT RACE RESULTS
# =============================================================================
def render_client_tab(manager, viz):
    """Render the Client Race Results tab."""
    st.markdown('<div class="tab-header">📊 Texas Governor Race Results</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Executive Summary:</strong> This section provides a high-level overview of Texas Governor
    race results, showing victory margins, party performance, and key trends over the 2010-2022 period.
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    metrics = viz.get_key_metrics()

    col1, col2, col3, col4, col5 = st.columns(5)

    display_metric_card(col1, "Avg Victory Margin",
                       format_percentage(metrics.get('avg_margin', 'N/A')),
                       "Republican advantage")

    display_metric_card(col2, "Closest Race",
                       str(metrics.get('closest_race', 'N/A')),
                       "Most competitive year")

    display_metric_card(col3, "Total Polls",
                       str(metrics.get('total_polls', 'N/A')),
                       "Analyzed")

    display_metric_card(col4, "Polling Error",
                       format_percentage(metrics.get('polling_error', 'N/A')),
                       "Mean absolute")

    display_metric_card(col5, "News Articles",
                       str(metrics.get('total_articles', 'N/A')),
                       "Coverage analyzed")

    st.markdown("<br>", unsafe_allow_html=True)

    # Election Results Section
    st.markdown('<div class="section-header">Election Results</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = viz.election_margin_trend()
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = viz.party_performance_comparison()
        st.plotly_chart(fig, use_container_width=True)

    # Results Table
    st.markdown('<div class="section-header">Election Results Summary</div>', unsafe_allow_html=True)

    margin_data = manager.all_results.get('elections', {}).get('margin_statistics', {}).get('by_year', {})
    if margin_data:
        results_df = pd.DataFrame([
            {
                'Year': year,
                'Winner': data.get('winner', 'N/A'),
                'Party': data.get('winner_party', 'N/A'),
                'Victory Margin': f"{data.get('margin_pct', 0):.1f}%"
            }
            for year, data in sorted(margin_data.items())
        ])
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Campaign Finance Section
    st.markdown('<div class="section-header">Campaign Finance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = viz.fundraising_by_year()
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = viz.party_fundraising_comparison()
        st.plotly_chart(fig, use_container_width=True)

    # Finance metrics
    finance = manager.all_results.get('campaign_finance', {})
    money_results = finance.get('money_vs_results', {})

    if money_results:
        st.markdown(f"""
        <div class="success-box">
        <strong>Money & Results:</strong> The top fundraiser won {money_results.get('money_win_rate', 'N/A')}%
        of races analyzed. Out of {money_results.get('cycles_analyzed', 0)} election cycles, the candidate
        who raised the most money won {money_results.get('top_fundraiser_won', 0)} times.
        </div>
        """, unsafe_allow_html=True)

    # Polling Section
    st.markdown('<div class="section-header">Polling Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = viz.polling_margin_by_year()
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = viz.polling_accuracy()
        st.plotly_chart(fig, use_container_width=True)

    # Polling accuracy note
    accuracy = manager.all_results.get('polling', {}).get('polling_accuracy', {}).get('overall', {})
    if accuracy:
        st.markdown(f"""
        <div class="warning-box">
        <strong>Polling Accuracy:</strong> Historical polling showed a mean absolute error of
        {accuracy.get('mean_absolute_error', 'N/A')} points. The systematic bias was:
        {accuracy.get('direction', 'Unknown')}.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: MODEL TEST RESULTS
# =============================================================================
def render_model_tab(manager, viz):
    """Render the Model Test Results tab."""
    st.markdown('<div class="tab-header">🔬 Statistical Model Results</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Model Overview:</strong> This section presents detailed statistical analysis including
    descriptive statistics, correlation analysis, and cross-dataset comparisons.
    </div>
    """, unsafe_allow_html=True)

    # Model Configuration
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Data Source</div>
            <div class="metric-value" style="font-size: 1.2rem;">ETL Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Analysis Period</div>
            <div class="metric-value" style="font-size: 1.2rem;">2010-2025</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Statistics Classes</div>
            <div class="metric-value" style="font-size: 1.2rem;">7</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Analysis Status</div>
            <div class="metric-value" style="font-size: 1.2rem; color: #4caf50;">Complete</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Statistics by Category
    st.markdown('<div class="section-header">Detailed Statistics</div>', unsafe_allow_html=True)

    # Create expandable sections for each statistics category
    with st.expander("📊 Election Statistics", expanded=True):
        elections = manager.all_results.get('elections', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Vote Statistics**")
            vote_stats = elections.get('vote_statistics', {}).get('overall', {})
            if vote_stats:
                st.json(vote_stats)

        with col2:
            st.markdown("**Margin Statistics**")
            margin_stats = elections.get('margin_statistics', {}).get('competitiveness', {})
            if margin_stats:
                st.json(margin_stats)

        st.markdown("**Historical Trends**")
        trends = elections.get('historical_trends', {})
        if trends:
            st.json(trends)

    with st.expander("💰 Campaign Finance Statistics"):
        finance = manager.all_results.get('campaign_finance', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Fundraising Overview**")
            fundraising = finance.get('fundraising_statistics', {}).get('overall', {})
            if fundraising:
                st.json(fundraising)

        with col2:
            st.markdown("**Spending Analysis**")
            spending = finance.get('spending_statistics', {})
            if spending:
                st.json(spending)

        # Expenditure chart
        fig = viz.expenditure_categories()
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📈 Polling Statistics"):
        polling = manager.all_results.get('polling', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Poll Summary**")
            poll_summary = polling.get('poll_summary', {})
            if poll_summary:
                st.json(poll_summary)

            fig = viz.pollster_activity()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Sample Size Analysis**")
            sample_stats = polling.get('sample_size_analysis', {})
            if sample_stats:
                st.json(sample_stats)

            st.markdown("**Trend Analysis**")
            trend_stats = polling.get('trend_analysis', {})
            if trend_stats.get('volatility_summary'):
                st.json(trend_stats['volatility_summary'])

    with st.expander("📰 News Coverage Statistics"):
        news = manager.all_results.get('news', {})

        col1, col2 = st.columns(2)

        with col1:
            fig = viz.news_coverage_by_year()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = viz.news_by_source()
            st.plotly_chart(fig, use_container_width=True)

        fig = viz.news_by_topic()
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("⚔️ Culture War Statistics"):
        culture_war = manager.all_results.get('culture_war', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Event Summary**")
            event_summary = culture_war.get('event_summary', {})
            if event_summary:
                st.json(event_summary)

            fig = viz.culture_war_events_by_year()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Political Leaning**")
            political = culture_war.get('political_leaning_analysis', {})
            if political:
                st.json({k: v for k, v in political.items() if not k.endswith('_distribution')})

            fig = viz.culture_war_political_leaning()
            st.plotly_chart(fig, use_container_width=True)

        fig = viz.culture_war_by_industry()
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📉 Market Statistics"):
        market = manager.all_results.get('market', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**VIX Analysis**")
            vix = market.get('vix_analysis', {})
            if vix:
                st.json({k: v for k, v in vix.items() if k != 'summary'})

            fig = viz.vix_summary()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**VIX Distribution**")
            if vix.get('summary'):
                st.json(vix['summary'])

            fig = viz.vix_distribution()
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🏛️ Macroeconomic Statistics"):
        macro = manager.all_results.get('macroeconomic', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Inflation Analysis**")
            inflation = macro.get('inflation_analysis', {})
            if inflation:
                st.json({k: v for k, v in inflation.items() if k != 'cpi_summary'})

            st.markdown("**GDP Analysis**")
            gdp = macro.get('gdp_analysis', {})
            if gdp:
                st.json({k: v for k, v in gdp.items() if k != 'growth_summary'})

        with col2:
            st.markdown("**Employment Analysis**")
            employment = macro.get('employment_analysis', {})
            if employment:
                st.json({k: v for k, v in employment.items() if k != 'unemployment_summary'})

            st.markdown("**Rates Analysis**")
            rates = macro.get('rates_analysis', {})
            if rates:
                st.json(rates)

        fig = viz.macro_indicators_summary()
        st.plotly_chart(fig, use_container_width=True)

    # Cross-Dataset Correlations
    st.markdown('<div class="section-header">Cross-Dataset Analysis</div>', unsafe_allow_html=True)

    fig = viz.integrated_cycle_comparison()
    st.plotly_chart(fig, use_container_width=True)

    correlations = manager.all_results.get('correlations', {})
    if correlations.get('polls_vs_results'):
        st.markdown("**Polls vs Results Analysis**")
        st.json(correlations['polls_vs_results'])


# =============================================================================
# TAB 3: ACADEMIC
# =============================================================================
def render_academic_tab(manager):
    """Render the Academic tab with raw data and code."""
    st.markdown('<div class="tab-header">🎓 Academic Resources</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>For Researchers:</strong> This section provides access to raw data, methodology documentation,
    code samples, and detailed data dictionaries suitable for academic review and replication studies.
    </div>
    """, unsafe_allow_html=True)

    # Sub-tabs within Academic
    academic_tabs = st.tabs(["📁 Raw Data", "📖 Methodology", "💻 Code", "📋 Data Dictionary"])

    # RAW DATA TAB
    with academic_tabs[0]:
        st.markdown("### Raw Data Access")

        raw_data = load_raw_data()

        if raw_data:
            # Dataset selector
            dataset_name = st.selectbox(
                "Select Dataset",
                options=list(raw_data.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )

            if dataset_name:
                df = raw_data[dataset_name]

                # Dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

                # Data preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head(100), use_container_width=True)

                # Column info
                st.markdown("#### Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(1)
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{dataset_name}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No raw data files found. Run the ETL pipeline first.")

    # METHODOLOGY TAB
    with academic_tabs[1]:
        st.markdown("### Methodology")

        st.markdown("""
        #### Research Design

        This study employs a multi-source data integration approach to analyze Texas Governor
        races from 2010-2024. The methodology combines quantitative analysis of electoral data
        with contextual factors including campaign finance, media coverage, and macroeconomic conditions.

        #### Data Collection

        **Primary Sources:**
        1. **Election Data**: Texas Secretary of State official results
        2. **Campaign Finance**: Federal Election Commission (FEC) and Texas Ethics Commission
        3. **Polling Data**: FiveThirtyEight aggregated polls with pollster ratings
        4. **News Coverage**: Guardian API and New York Times API
        5. **Market Data**: Yahoo Finance (VIX, stock prices), Kenneth French Data Library (Fama-French factors)
        6. **Economic Data**: Federal Reserve Economic Data (FRED)

        #### Statistical Methods

        **Descriptive Statistics:**
        - Central tendency measures (mean, median, mode)
        - Dispersion measures (standard deviation, IQR, range)
        - Distribution characteristics (skewness, kurtosis)

        **Time Series Analysis:**
        - Trend analysis with linear regression
        - Volatility measurement
        - Change point detection

        **Correlation Analysis:**
        - Cross-dataset correlations
        - Money-votes relationship
        - Polling accuracy assessment

        #### Data Processing Pipeline

        The ETL (Extract, Transform, Load) pipeline follows these steps:

        1. **Extract**: Retrieve data from APIs and local sources
        2. **Transform**: Clean, normalize, and structure data
        3. **Load**: Store in Snowflake database or local CSV files

        #### Limitations

        - Historical data availability varies by election cycle
        - Polling data subject to house effects and methodological differences
        - Campaign finance data may have reporting delays
        - News coverage analysis limited to English-language sources

        #### Reproducibility

        All code is available in the project repository. Data extraction can be replicated
        using the provided API keys and ETL pipeline scripts.
        """)

        # Export methodology as PDF-ready markdown
        methodology_text = """
# Texas Governor Race Analysis - Methodology

## Research Design
Multi-source data integration approach analyzing Texas Governor races (2010-2024).

## Data Sources
1. Texas Secretary of State - Election Results
2. FEC/Texas Ethics Commission - Campaign Finance
3. FiveThirtyEight - Polling Data
4. Guardian/NYT APIs - News Coverage
5. Yahoo Finance/FRED - Market & Economic Data

## Statistical Methods
- Descriptive statistics (mean, std, quartiles)
- Time series trend analysis
- Cross-dataset correlation analysis

## Data Pipeline
ETL process: Extract -> Transform -> Load (Snowflake/CSV)
        """

        st.download_button(
            label="📥 Download Methodology (Markdown)",
            data=methodology_text,
            file_name="methodology.md",
            mime="text/markdown"
        )

    # CODE TAB
    with academic_tabs[2]:
        st.markdown("### Code Samples")

        st.markdown("#### Project Structure")
        st.code("""
texas-governor-analysis/
├── clean.py           # Data loading and cleaning functions
├── ETL.py             # Extract, Transform, Load pipeline
├── database.py        # Snowflake database operations
├── Model.py           # Statistical analysis classes
├── visualizations.py  # Plotly visualization functions
├── streamlit_app.py   # This dashboard
├── requirements.txt   # Python dependencies
├── .env               # API keys and credentials
└── data/              # Raw and processed data
        """, language="text")

        st.markdown("#### Key Code Snippets")

        with st.expander("Loading Election Data"):
            st.code("""
from Model import StatisticalModelManager

# Initialize the model manager
manager = StatisticalModelManager(use_database=False)
manager.initialize()

# Run all statistics
results = manager.run_all_statistics()

# Access election statistics
election_results = results['elections']
print(f"Average margin: {election_results['margin_statistics']['competitiveness']['avg_margin']}%")
            """, language="python")

        with st.expander("Running the ETL Pipeline"):
            st.code("""
from ETL import ETLPipeline

# Initialize pipeline
pipeline = ETLPipeline(
    start_year=2010,
    end_year=2025
)

# Run full pipeline
pipeline.run(
    extract=True,
    transform=True,
    load=True
)

# Access transformed data
election_df = pipeline.transformed_data['election_results']
finance_df = pipeline.transformed_data['finance_summary']
polls_df = pipeline.transformed_data['polls']
            """, language="python")

        with st.expander("Creating Visualizations"):
            st.code("""
from Model import StatisticalModelManager
from visualizations import Visualizer

# Load data
manager = StatisticalModelManager()
manager.initialize()
manager.run_all_statistics()

# Create visualizer
viz = Visualizer(model_manager=manager)

# Generate charts
margin_chart = viz.election_margin_trend()
finance_chart = viz.fundraising_by_year()
polling_chart = viz.polling_accuracy()

# Display in Streamlit
import streamlit as st
st.plotly_chart(margin_chart)
            """, language="python")

        with st.expander("Database Queries (Snowflake)"):
            st.code("""
from database import DatabaseManager

# Connect to Snowflake
db = DatabaseManager()
db.connect()

# Run queries
election_results = db.run_query('''
    SELECT
        ELECTION_YEAR,
        CANDIDATE_NAME,
        PARTY,
        VOTES,
        VOTE_PERCENTAGE
    FROM ELECTION_RESULTS_STATEWIDE
    ORDER BY ELECTION_YEAR, VOTES DESC
''')

# Load to Pandas
import pandas as pd
df = pd.DataFrame(election_results)

# Close connection
db.close()
            """, language="python")

        with st.expander("Statistical Analysis Classes"):
            st.code("""
from Model import (
    ElectionStatistics,
    CampaignFinanceStatistics,
    PollingStatistics,
    NewsStatistics,
    CultureWarStatistics,
    MarketStatistics,
    MacroeconomicStatistics
)

# Each class follows the same pattern:
class ExampleStatistics(DescriptiveStatistics):
    def __init__(self, db_manager=None, data=None):
        self.db = db_manager
        self.data = data or {}
        self.results = {}

    def load_data(self):
        '''Load data from database or files'''
        pass

    def compute_all(self):
        '''Run all statistical computations'''
        return {
            'summary': self._compute_summary(),
            'trends': self._compute_trends(),
            # ... other analyses
        }
            """, language="python")

        # Full results JSON download
        st.markdown("#### Export Full Results")

        results_json = json.dumps(manager.all_results, indent=2, default=str)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name="analysis_results.json",
            mime="application/json"
        )

    # DATA DICTIONARY TAB
    with academic_tabs[3]:
        st.markdown("### Data Dictionary")

        st.markdown("""
        The data dictionary defines all datasets, their sources, and field definitions
        used in this analysis.
        """)

        # Display DATA_DICTIONARY from ETL
        for category, info in DATA_DICTIONARY.items():
            with st.expander(f"📊 {category.replace('_', ' ').title()}", expanded=False):
                st.markdown(f"**Description:** {info.get('description', 'N/A')}")
                st.markdown(f"**Source:** {info.get('source', 'N/A')}")

                if 'tables' in info:
                    st.markdown("**Tables:**")
                    for table_name, table_info in info['tables'].items():
                        st.markdown(f"- **{table_name}**: {table_info.get('description', 'N/A')}")

                        if 'columns' in table_info:
                            cols_df = pd.DataFrame([
                                {'Column': col, 'Type': details.get('type', 'N/A'),
                                 'Description': details.get('description', 'N/A')}
                                for col, details in table_info['columns'].items()
                            ])
                            st.dataframe(cols_df, use_container_width=True, hide_index=True)

        # Export data dictionary
        dict_json = json.dumps(DATA_DICTIONARY, indent=2)
        st.download_button(
            label="📥 Download Data Dictionary (JSON)",
            data=dict_json,
            file_name="data_dictionary.json",
            mime="application/json"
        )


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()

    # Main header
    st.markdown('<div class="main-header">🗳️ Texas Governor Race Analysis</div>',
                unsafe_allow_html=True)

    # Load data with spinner
    with st.spinner("Loading statistical model..."):
        try:
            manager = load_model_manager()
            viz = Visualizer(model_manager=manager)
            data_loaded = True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            data_loaded = False

    if not data_loaded:
        st.warning("Could not load data. Please check the data sources and try again.")
        return

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Client Race Results",
        "🔬 Model Test Results",
        "🎓 Academic"
    ])

    with tab1:
        render_client_tab(manager, viz)

    with tab2:
        render_model_tab(manager, viz)

    with tab3:
        render_academic_tab(manager)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Texas Governor Race Analysis Dashboard | Roseboro Foundation</p>
        <p>Data Sources: Texas SoS, FEC, FiveThirtyEight, Guardian, NYT, Yahoo Finance, FRED</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
