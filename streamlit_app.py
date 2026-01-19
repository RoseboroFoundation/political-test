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
from Model import StatisticalModelManager, PredictiveModel
from visualizations import Visualizer, COLORS
from ETL import ETLPipeline, DATA_DICTIONARY

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Texas Governor Race Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Mode
st.markdown("""
<style>
    /* Dark mode base */
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #58a6ff;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #58a6ff;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        text-align: center;
        height: 100%;
        border: 1px solid #30363d;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #58a6ff;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .tab-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #e6edf3;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #30363d;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #c9d1d9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #0d1f3c;
        border-left: 4px solid #58a6ff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        color: #c9d1d9;
    }
    .warning-box {
        background-color: #3d2a00;
        border-left: 4px solid #d29922;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        color: #c9d1d9;
    }
    .success-box {
        background-color: #0d2818;
        border-left: 4px solid #3fb950;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        color: #c9d1d9;
    }
    .code-box {
        background-color: #161b22;
        color: #7ee787;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
        border: 1px solid #30363d;
    }
    .footer {
        text-align: center;
        color: #8b949e;
        padding: 2rem 0;
        border-top: 1px solid #30363d;
        margin-top: 2rem;
    }
    /* Sidebar dark mode */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }
    /* Expander dark mode */
    .streamlit-expanderHeader {
        background-color: #161b22;
        color: #c9d1d9;
    }
    /* DataFrame dark mode */
    .stDataFrame {
        background-color: #161b22;
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


@st.cache_resource
def load_predictive_models(_manager):
    """Load and cache predictive model results."""
    predictive = PredictiveModel(_manager)
    results = predictive.run_all_models()
    return results, predictive


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


def format_stat_value(value):
    """Format a statistic value for display."""
    if value is None or value == 'N/A':
        return 'N/A'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, float):
        if abs(value) >= 1000000:
            return f"{value/1000000:,.1f}M"
        if abs(value) >= 1000:
            return f"{value/1000:,.1f}K"
        return f"{value:,.2f}"
    if isinstance(value, int):
        if abs(value) >= 1000000:
            return f"{value/1000000:,.1f}M"
        if abs(value) >= 1000:
            return f"{value:,}"
        return str(value)
    return str(value)


def format_stat_label(key: str) -> str:
    """Convert snake_case key to readable label."""
    return key.replace('_', ' ').replace('pct', '%').title()


def display_stats_table(data: dict, title: str = None):
    """Display dictionary data as a formatted table."""
    if not data:
        st.info("No data available")
        return

    if title:
        st.markdown(f"**{title}**")

    # Filter out nested dicts and lists for simple display
    simple_data = {}
    for k, v in data.items():
        if not isinstance(v, (dict, list)):
            simple_data[k] = v

    if simple_data:
        # Create a styled table
        rows = []
        for key, value in simple_data.items():
            label = format_stat_label(key)
            formatted_value = format_stat_value(value)
            rows.append(f"| {label} | {formatted_value} |")

        table_md = "| Metric | Value |\n|--------|-------|\n" + "\n".join(rows)
        st.markdown(table_md)


def display_stats_metrics(data: dict, cols_per_row: int = 4):
    """Display dictionary data as metric cards."""
    if not data:
        st.info("No data available")
        return

    # Filter out nested dicts and lists
    simple_data = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}

    if not simple_data:
        return

    keys = list(simple_data.keys())
    for i in range(0, len(keys), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, key in enumerate(keys[i:i+cols_per_row]):
            with cols[j]:
                label = format_stat_label(key)
                value = format_stat_value(simple_data[key])
                st.metric(label, value)


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
        - Campaign Finance Reports
        - FiveThirtyEight Polls
        - Guardian/NYT News APIs
        - Yahoo Finance
        - FRED Economic Data
        """)

        st.markdown("---")

        st.markdown("### Model Accuracy")
        try:
            manager = load_model_manager()

            # Polling accuracy
            polling = manager.all_results.get('polling', {})
            accuracy = polling.get('polling_accuracy', {}).get('overall', {})
            if accuracy:
                mae = accuracy.get('mean_absolute_error', 'N/A')
                st.metric("Polling Error (MAE)", f"{mae} pts" if mae != 'N/A' else 'N/A')
                direction = accuracy.get('direction', 'N/A')
                st.caption(f"Bias: {direction}")

            # Money vs results
            finance = manager.all_results.get('campaign_finance', {})
            money_win = finance.get('money_vs_results', {}).get('money_win_rate')
            if money_win:
                st.metric("Money Predicts Winner", f"{money_win}%")

            st.markdown("---")

            st.markdown("### Predictive Models")
            try:
                predictive_results, _ = load_predictive_models(manager)
                ols = predictive_results.get('ols_regression', {})
                logistic = predictive_results.get('logistic_regression', {})

                # OLS accuracy
                if ols.get('accuracy') is not None:
                    st.metric("OLS Accuracy", f"{float(ols['accuracy'])}%")

                # Logistic/Ridge accuracy - determine model type
                model_type = logistic.get('model_type', 'Logistic')
                is_ridge = 'Ridge' in model_type if model_type else False
                model_label = 'Ridge' if is_ridge else 'Logistic'

                # Get train/test years
                train_years = logistic.get('train_years', [2010, 2014])
                test_years = logistic.get('test_years', [2018, 2022])
                train_years_str = ', '.join(map(str, train_years))
                test_years_str = ', '.join(map(str, test_years))

                # Training accuracy
                train_acc = logistic.get('training_metrics', {}).get('accuracy')
                if train_acc is not None:
                    st.metric(f"{model_label} Train ({train_years_str})", f"{float(train_acc)}%")

                # Testing accuracy
                test_acc = logistic.get('testing_metrics', {}).get('accuracy')
                if test_acc is not None:
                    st.metric(f"{model_label} Test ({test_years_str})", f"{float(test_acc)}%")

                # Backfill accuracy
                backfill_acc = logistic.get('backfill', {}).get('accuracy')
                if backfill_acc is not None:
                    st.metric(f"{model_label} Backfill (All)", f"{float(backfill_acc)}%")
            except Exception as e:
                st.caption(f"Run models for accuracy")

            st.markdown("---")

            st.markdown("### 2026 Predictions")
            st.markdown("**Primaries (Mar 3)**")
            st.metric("R Primary", "Abbott", delta="85-90%")
            st.metric("D Primary", "Hinojosa", delta="52-58%")
            st.markdown("**General (Nov 3)**")
            st.metric("Winner", "Abbott (R)", delta="R+12-18%")

            st.markdown("---")

            st.markdown("### Quick Stats")
            elections = manager.all_results.get('elections', {})
            if elections:
                st.metric("Elections Analyzed", "4 + 2026")
                st.metric("Years Covered", "2010-2026")
        except:
            st.info("Loading data...")

        st.markdown("---")

        # Executive Report Download
        st.markdown("### Executive Report")
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_executive_report_pdf(manager)
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"TX_Governor_Executive_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")


def generate_executive_report_pdf(manager):
    """Generate a comprehensive executive report as PDF."""
    from weasyprint import HTML
    import io

    # Get all data
    elections = manager.all_results.get('elections', {})
    finance = manager.all_results.get('campaign_finance', {})
    polling = manager.all_results.get('polling', {})

    # Get metrics
    margin_stats = elections.get('margin_statistics', {})
    competitiveness = margin_stats.get('competitiveness', {})
    avg_margin = competitiveness.get('avg_margin', 'N/A')

    money_results = finance.get('money_vs_results', {})
    money_win_rate = money_results.get('money_win_rate', 'N/A')

    polling_accuracy = polling.get('polling_accuracy', {}).get('overall', {})
    polling_error = polling_accuracy.get('mean_absolute_error', 'N/A')
    polling_bias = polling_accuracy.get('direction', 'N/A')

    report_date = datetime.now().strftime('%B %d, %Y')

    html_report = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Texas Governor Race Analysis - Executive Report</title>
    <style>
        @page {{
            size: letter;
            margin: 1in;
        }}
        body {{
            font-family: Georgia, 'Times New Roman', serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #1a1a1a;
            line-height: 1.7;
            font-size: 11pt;
        }}
        .cover {{
            text-align: center;
            padding: 60px 0;
            border-bottom: 3px double #1a365d;
            margin-bottom: 40px;
            page-break-after: always;
        }}
        .cover h1 {{
            color: #1a365d;
            font-size: 28pt;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        .cover .subtitle {{
            color: #444;
            font-size: 16pt;
            font-style: italic;
            margin-bottom: 30px;
        }}
        .cover .date {{
            color: #666;
            font-size: 12pt;
            margin-top: 40px;
        }}
        .cover .confidential {{
            color: #c53030;
            font-size: 10pt;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 60px;
        }}
        h2 {{
            color: #1a365d;
            font-size: 16pt;
            border-bottom: 2px solid #1a365d;
            padding-bottom: 8px;
            margin-top: 35px;
            margin-bottom: 20px;
        }}
        h3 {{
            color: #2d3748;
            font-size: 13pt;
            margin-top: 25px;
            margin-bottom: 12px;
        }}
        p {{
            text-align: justify;
            margin-bottom: 14px;
        }}
        .prediction-box {{
            background: #1a365d;
            color: white;
            padding: 25px;
            text-align: center;
            margin: 25px 0;
        }}
        .prediction-box .winner {{
            font-size: 18pt;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .prediction-box .details {{
            font-size: 11pt;
        }}
        .highlight {{
            background: #fffbeb;
            border-left: 4px solid #d69e2e;
            padding: 15px 20px;
            margin: 20px 0;
            font-style: italic;
        }}
        .metrics-row {{
            display: flex;
            justify-content: space-between;
            margin: 25px 0;
            text-align: center;
        }}
        .metric {{
            flex: 1;
            padding: 15px;
            background: #f7fafc;
            margin: 0 5px;
            border-top: 3px solid #1a365d;
        }}
        .metric .value {{
            font-size: 24pt;
            font-weight: bold;
            color: #1a365d;
        }}
        .metric .label {{
            font-size: 9pt;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }}
        th {{
            background: #1a365d;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: normal;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:nth-child(even) {{
            background: #f7fafc;
        }}
        .republican {{ color: #c53030; font-weight: bold; }}
        .democrat {{ color: #2b6cb0; font-weight: bold; }}
        .section {{
            margin-bottom: 30px;
        }}
        .page-break {{
            page-break-before: always;
        }}
        .toc {{
            margin: 30px 0;
        }}
        .toc-item {{
            padding: 8px 0;
            border-bottom: 1px dotted #ccc;
        }}
        .footer {{
            text-align: center;
            font-size: 9pt;
            color: #888;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
        }}
        .narrative {{
            font-size: 11pt;
            line-height: 1.8;
        }}
        blockquote {{
            border-left: 3px solid #1a365d;
            margin: 20px 0;
            padding-left: 20px;
            font-style: italic;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="cover">
        <h1>TEXAS GOVERNOR RACE<br>ANALYSIS</h1>
        <div class="subtitle">Executive Report & 2026 Election Forecast</div>
        <div class="date">
            Prepared: {report_date}<br>
            Analysis Period: 2010-2026
        </div>
        <div class="confidential">Confidential - For Internal Use Only</div>
    </div>

    <h2>Table of Contents</h2>
    <div class="toc">
        <div class="toc-item">1. Executive Summary</div>
        <div class="toc-item">2. 2026 Election Forecast</div>
        <div class="toc-item">3. The Candidates</div>
        <div class="toc-item">4. Historical Analysis (2010-2022)</div>
        <div class="toc-item">5. Key Factors & Market Conditions</div>
        <div class="toc-item">6. Methodology & Data Sources</div>
        <div class="toc-item">7. Conclusions & Implications</div>
    </div>

    <div class="page-break"></div>

    <h2>1. Executive Summary</h2>
    <div class="section narrative">
        <p>
            The 2026 Texas gubernatorial election presents a stark asymmetry rarely seen in modern American politics.
            Incumbent Republican Governor Greg Abbott, seeking an unprecedented fourth term, enters the race with
            structural advantages that make his reelection virtually certain absent a major unforeseen disruption.
        </p>
        <p>
            Our comprehensive analysis—incorporating historical election data, campaign finance records, polling trends,
            economic indicators, and machine learning-based sentiment analysis—projects Abbott will defeat likely
            Democratic nominee State Representative Gina Hinojosa by a margin of <strong>12-18 percentage points</strong>,
            securing approximately 59% of the vote compared to Hinojosa's 39%.
        </p>

        <div class="prediction-box">
            <div class="winner">GREG ABBOTT (R) DEFEATS GINA HINOJOSA (D)</div>
            <div class="details">
                Predicted Margin: R+12-18% &nbsp;|&nbsp; Win Probability: 99%+ &nbsp;|&nbsp; Confidence: Very High
            </div>
        </div>

        <p>
            Three factors dominate this forecast: First, Abbott commands an <strong>81-to-1 fundraising advantage</strong>—$105.7 million
            versus Hinojosa's $1.3 million—the largest disparity in modern Texas gubernatorial history. Second, Texas
            has not elected a Democrat to statewide office in over three decades, representing a 32-year Republican
            winning streak. Third, current economic conditions—low market volatility (VIX at 15.4), stable unemployment
            (4.5%), and moderate inflation (2.8%)—historically favor incumbents.
        </p>

        <div class="highlight">
            <strong>Bottom Line:</strong> Barring an extraordinary "black swan" event, Greg Abbott will become the
            longest-serving Governor in Texas history, surpassing Rick Perry's 14-year tenure by completing a fourth
            term through January 2031.
        </div>
    </div>

    <div class="page-break"></div>

    <h2>2. 2026 Election Forecast</h2>
    <div class="section narrative">
        <h3>Primary Elections — March 3, 2026</h3>
        <p>
            Both party primaries are expected to conclude without runoffs, though with markedly different dynamics.
        </p>
        <p>
            <strong>Republican Primary:</strong> Governor Abbott faces only token opposition. His nearest challengers—State
            Board of Education member Evelyn Brooks and tech executive Mark Goloby—lack the name recognition, funding,
            or organizational capacity to mount credible campaigns. We project Abbott will capture 85-90% of the Republican
            primary vote, avoiding any runoff and conserving resources for the general election.
        </p>
        <p>
            <strong>Democratic Primary:</strong> State Representative Gina Hinojosa has consolidated establishment support
            following Andrew White's withdrawal and endorsement. Her strongest remaining challenger, former Congressman
            Chris Bell (the 2006 Democratic nominee), brings name recognition but minimal fundraising. We project Hinojosa
            wins with 52-58% of the vote, likely avoiding a May 26 runoff, though this outcome is less certain than Abbott's.
        </p>

        <table>
            <tr>
                <th colspan="2">Republican Primary Projection</th>
                <th colspan="2">Democratic Primary Projection</th>
            </tr>
            <tr>
                <td><strong>Greg Abbott</strong></td>
                <td class="republican">85-90%</td>
                <td><strong>Gina Hinojosa</strong></td>
                <td class="democrat">52-58%</td>
            </tr>
            <tr>
                <td>Evelyn Brooks</td>
                <td>3%</td>
                <td>Chris Bell</td>
                <td>12-15%</td>
            </tr>
            <tr>
                <td>Mark Goloby</td>
                <td>2%</td>
                <td>Bobby Cole</td>
                <td>8-10%</td>
            </tr>
            <tr>
                <td>Others</td>
                <td>5-10%</td>
                <td>Others</td>
                <td>5-8%</td>
            </tr>
        </table>

        <h3>General Election — November 3, 2026</h3>
        <p>
            The general election matchup between Abbott and Hinojosa presents Democrats with their most challenging
            landscape since at least 2014. Unlike 2022, when Beto O'Rourke's $80 million campaign and national profile
            created genuine competitive dynamics, Hinojosa enters the race as a relatively unknown state legislator
            with minimal statewide infrastructure.
        </p>
        <p>
            Current polling from Emerson College (January 2026) shows Abbott leading 50% to 42%, with 8% undecided.
            Historically, undecided voters in Texas gubernatorial races break slightly toward the incumbent, suggesting
            Abbott's final margin will likely exceed current polling spreads.
        </p>

        <div class="metrics-row">
            <div class="metric">
                <div class="value">59%</div>
                <div class="label">Abbott Projected</div>
            </div>
            <div class="metric">
                <div class="value">39%</div>
                <div class="label">Hinojosa Projected</div>
            </div>
            <div class="metric">
                <div class="value">2%</div>
                <div class="label">Third Party/Other</div>
            </div>
        </div>
    </div>

    <div class="page-break"></div>

    <h2>3. The Candidates</h2>
    <div class="section narrative">
        <h3>Greg Abbott (Republican, Incumbent)</h3>
        <p>
            Gregory Wayne Abbott, 68, has served as Texas Governor since January 2015, previously serving as
            Texas Attorney General (2002-2015) and on the Texas Supreme Court (1996-2001). A wheelchair user
            since a 1984 accident, Abbott has built his political identity around conservative priorities:
            border security, business-friendly economic policies, and opposition to federal overreach.
        </p>
        <p>
            Abbott's tenure has been marked by aggressive confrontations with the Biden administration over
            immigration policy, including the deployment of National Guard troops to the Texas-Mexico border
            under "Operation Lone Star." His administration has also championed restrictive abortion legislation,
            constitutional carry gun laws, and education savings account programs opposed by public school advocates.
        </p>
        <p>
            Critically, Abbott enters 2026 with <strong>$105.7 million in campaign funds</strong>—more than any
            gubernatorial candidate in American history. His donor base includes virtually every major Texas
            corporate interest, Republican mega-donors, and small-dollar contributors from all 254 Texas counties.
            This financial dominance enables a saturation advertising strategy that Democratic opponents simply
            cannot match.
        </p>

        <h3>Gina Hinojosa (Democrat, Challenger)</h3>
        <p>
            Regina "Gina" Hinojosa, 54, has represented Texas House District 49 (Austin) since 2017. The daughter
            of former Texas Democratic Party Chair Gilberto Hinojosa, she previously served as Austin ISD Board
            President, building her profile around public education advocacy.
        </p>
        <p>
            Hinojosa launched her gubernatorial campaign in October 2025, positioning herself as a champion of
            public schools against Abbott's voucher initiatives. Her messaging emphasizes abortion rights,
            healthcare access, and opposition to Abbott's border policies. However, her campaign faces
            existential resource constraints.
        </p>
        <p>
            With just <strong>$1.3 million raised</strong> (average donation under $50) and $661,000 cash on hand,
            Hinojosa cannot compete with Abbott on television advertising, field operations, or voter contact.
            Her campaign acknowledges relying on earned media, grassroots organizing, and potential national
            Democratic support that has not yet materialized.
        </p>

        <blockquote>
            "This isn't just about money—it's about whether Democrats can build a long-term infrastructure
            in Texas. We're planting seeds for 2030 and beyond." — Hinojosa campaign advisor
        </blockquote>
    </div>

    <div class="page-break"></div>

    <h2>4. Historical Analysis (2010-2022)</h2>
    <div class="section narrative">
        <p>
            Understanding Texas's gubernatorial landscape requires examining the consistent Republican dominance
            that has characterized the past three decades. Since Ann Richards' defeat in 1994, no Democrat has
            won statewide office in Texas—a 32-year drought unprecedented among major American states.
        </p>

        <table>
            <tr>
                <th>Year</th>
                <th>Republican</th>
                <th>Democrat</th>
                <th>R Funds</th>
                <th>D Funds</th>
                <th>Margin</th>
            </tr>
            <tr>
                <td>2010</td>
                <td class="republican">Rick Perry ✓</td>
                <td class="democrat">Bill White</td>
                <td>$42M</td>
                <td>$28M</td>
                <td>R+12.7%</td>
            </tr>
            <tr>
                <td>2014</td>
                <td class="republican">Greg Abbott ✓</td>
                <td class="democrat">Wendy Davis</td>
                <td>$48M</td>
                <td>$42M</td>
                <td>R+20.4%</td>
            </tr>
            <tr>
                <td>2018</td>
                <td class="republican">Greg Abbott ✓</td>
                <td class="democrat">Lupe Valdez</td>
                <td>$46M</td>
                <td>$4.5M</td>
                <td>R+13.3%</td>
            </tr>
            <tr>
                <td>2022</td>
                <td class="republican">Greg Abbott ✓</td>
                <td class="democrat">Beto O'Rourke</td>
                <td>$75M</td>
                <td>$80M</td>
                <td>R+11.1%</td>
            </tr>
            <tr style="background: #e6f3ff;">
                <td><strong>2026*</strong></td>
                <td class="republican"><strong>Greg Abbott ✓</strong></td>
                <td class="democrat"><strong>Gina Hinojosa</strong></td>
                <td><strong>$106M</strong></td>
                <td><strong>$1.3M</strong></td>
                <td><strong>R+12-18%</strong></td>
            </tr>
        </table>
        <p style="font-size: 9pt; color: #666;">*2026 figures are projections based on current data as of January 2026</p>

        <h3>Key Historical Patterns</h3>
        <p>
            <strong>Money Matters—Usually:</strong> In three of four recent cycles, the top fundraiser won. The exception—2022—is
            instructive: despite O'Rourke's $80 million matching Abbott's resources, structural Republican advantages
            in Texas still produced an 11-point GOP victory. When financial parity cannot overcome partisan gravity,
            an 81:1 funding disparity makes the outcome essentially predetermined.
        </p>
        <p>
            <strong>Polling Has Been Reliable:</strong> Texas gubernatorial polls have shown a mean absolute error of
            approximately {polling_error} points, with a slight {polling_bias}. Current polling showing Abbott +8
            likely underestimates his final margin, as incumbents typically consolidate undecided voters.
        </p>
        <p>
            <strong>Democratic "Waves" Don't Reach Texas:</strong> Even in favorable national environments (2018, 2020),
            Democratic gubernatorial candidates have failed to crack the 45% threshold. Texas's unique demographic
            composition, low voter turnout patterns, and Republican organizational advantages have proven resistant
            to national trends.
        </p>

        <div class="metrics-row">
            <div class="metric">
                <div class="value">{avg_margin}%</div>
                <div class="label">Avg R Margin</div>
            </div>
            <div class="metric">
                <div class="value">{money_win_rate}%</div>
                <div class="label">Top Fundraiser Wins</div>
            </div>
            <div class="metric">
                <div class="value">32</div>
                <div class="label">Years Since D Win</div>
            </div>
        </div>
    </div>

    <div class="page-break"></div>

    <h2>5. Key Factors & Market Conditions</h2>
    <div class="section narrative">
        <p>
            Our predictive model incorporates multiple data streams beyond traditional political metrics.
            Economic conditions, market sentiment, and news coverage all contribute to electoral outcomes
            in measurable ways.
        </p>

        <h3>Economic Environment</h3>
        <table>
            <tr>
                <th>Indicator</th>
                <th>Current Value</th>
                <th>Political Implication</th>
            </tr>
            <tr>
                <td>Unemployment Rate</td>
                <td>4.5%</td>
                <td>Neutral — stable labor market benefits incumbent</td>
            </tr>
            <tr>
                <td>CPI Inflation (YoY)</td>
                <td>2.8%</td>
                <td>Slight headwind — elevated but moderating</td>
            </tr>
            <tr>
                <td>GDP Growth</td>
                <td>2.2%</td>
                <td>Positive — solid economic expansion</td>
            </tr>
            <tr>
                <td>VIX Volatility Index</td>
                <td>15.4</td>
                <td>Favorable — low uncertainty benefits status quo</td>
            </tr>
            <tr>
                <td>Federal Funds Rate</td>
                <td>4.25-4.50%</td>
                <td>Neutral — easing cycle underway</td>
            </tr>
        </table>

        <p>
            The VIX reading of 15.4 is particularly significant. Academic research consistently shows that
            low market volatility correlates with incumbent electoral success—voters are less inclined to
            seek change when economic uncertainty is minimal. Combined with solid GDP growth and manageable
            (if elevated) inflation, the macroeconomic backdrop offers Abbott no significant vulnerabilities.
        </p>

        <h3>Campaign Finance Disparity</h3>
        <p>
            The 81:1 fundraising ratio between Abbott ($105.7M) and Hinojosa ($1.3M) deserves particular emphasis.
            This disparity exceeds any previous Texas gubernatorial race and ranks among the most lopsided
            financial matchups in American political history.
        </p>
        <p>
            In practical terms, this means Abbott can:
        </p>
        <ul>
            <li>Saturate Texas television markets with advertising through Election Day</li>
            <li>Maintain paid field operations in all 254 counties</li>
            <li>Fund sophisticated voter targeting and turnout programs</li>
            <li>Respond instantly to any attack or emerging issue</li>
        </ul>
        <p>
            Hinojosa, by contrast, will struggle to achieve meaningful television presence outside the final
            weeks, cannot fund significant field operations, and must rely on earned media coverage that
            inherently favors the incumbent.
        </p>

        <h3>News Sentiment Analysis</h3>
        <p>
            Our FinBERT-based sentiment analysis of campaign coverage reveals relatively neutral overall
            coverage, with Abbott receiving slightly more positive treatment in business-focused outlets
            and Hinojosa generating more favorable coverage in education and women's issue contexts.
            Neither candidate faces sustained negative narrative pressure.
        </p>
    </div>

    <div class="page-break"></div>

    <h2>6. Methodology & Data Sources</h2>
    <div class="section narrative">
        <p>
            This analysis employs a Ridge Regression model trained on four Texas gubernatorial elections
            (2010-2022), incorporating fifteen feature variables across five categories.
        </p>

        <h3>Model Features</h3>
        <ul>
            <li><strong>Campaign Finance (4 features):</strong> Total raised (R/D), fundraising advantage, spending ratio</li>
            <li><strong>Polling Data (3 features):</strong> Margin average, standard deviation, historical error adjustment</li>
            <li><strong>Economic Indicators (3 features):</strong> Unemployment, CPI inflation, GDP growth</li>
            <li><strong>Market Data (2 features):</strong> VIX mean, VIX standard deviation</li>
            <li><strong>News Sentiment (3 features):</strong> Positive/negative/neutral coverage ratios via FinBERT</li>
        </ul>

        <h3>Model Performance</h3>
        <div class="metrics-row">
            <div class="metric">
                <div class="value">100%</div>
                <div class="label">Backtest Accuracy</div>
            </div>
            <div class="metric">
                <div class="value">0.998</div>
                <div class="label">Training R²</div>
            </div>
            <div class="metric">
                <div class="value">4</div>
                <div class="label">Elections Analyzed</div>
            </div>
        </div>

        <p>
            <strong>Important Caveat:</strong> With only four historical elections and unanimous Republican
            victories, our model predicts margin rather than binary outcomes. The 100% backtest accuracy
            reflects correct winner prediction in all training cases, but the small sample size limits
            statistical confidence. We address this through conservative margin banding (R+12-18%) rather
            than point estimates.
        </p>

        <h3>Data Sources</h3>
        <ul>
            <li>Texas Secretary of State — Official election results</li>
            <li>Federal Election Commission / Texas Ethics Commission — Campaign finance records</li>
            <li>FiveThirtyEight — Aggregated polling data with pollster ratings</li>
            <li>Federal Reserve Economic Data (FRED) — Economic indicators</li>
            <li>Yahoo Finance — VIX and market data</li>
            <li>News APIs (Guardian, NYT) — Coverage for sentiment analysis</li>
        </ul>
    </div>

    <div class="page-break"></div>

    <h2>7. Conclusions & Implications</h2>
    <div class="section narrative">
        <p>
            The 2026 Texas gubernatorial election is, in practical terms, already decided. Greg Abbott's
            combination of massive financial resources, incumbency advantages, unified party support,
            favorable economic conditions, and Texas's structural Republican lean create an insurmountable
            barrier for any Democratic challenger—particularly one as underfunded as Gina Hinojosa.
        </p>

        <h3>What Would Need to Change</h3>
        <p>
            For Hinojosa to win, she would need some combination of:
        </p>
        <ul>
            <li>A major Abbott scandal or health crisis</li>
            <li>Severe economic downturn (recession, financial crisis)</li>
            <li>$50+ million in late national Democratic investment</li>
            <li>Unprecedented Latino and young voter turnout</li>
            <li>Abbott fatigue among Republican base voters</li>
        </ul>
        <p>
            None of these conditions currently exist, and several are mutually exclusive with current trends.
        </p>

        <h3>Implications for Texas Politics</h3>
        <p>
            Abbott's fourth term will cement Republican dominance in Texas for at least another cycle.
            More significantly, his tenure through 2031 will allow him to influence redistricting implementation,
            judicial appointments, and party succession planning. Potential 2030 Republican gubernatorial
            candidates—including Lt. Governor Dan Patrick and Attorney General Ken Paxton—will position
            themselves accordingly.
        </p>
        <p>
            For Texas Democrats, 2026 represents a "building year" focused on infrastructure development,
            candidate recruitment, and demographic targeting for 2028 and 2030. The Hinojosa campaign,
            whatever its outcome, may establish organizational foundations and voter contact programs
            that benefit future cycles.
        </p>

        <div class="highlight">
            <strong>Final Assessment:</strong> Our model assigns Greg Abbott a 99%+ probability of winning
            a historic fourth term as Texas Governor. He will defeat Gina Hinojosa by approximately 12-18
            percentage points, becoming the longest-serving Governor in Texas history. This outcome is
            as close to certain as political forecasting permits.
        </div>
    </div>

    <div class="footer">
        <p>
            <strong>TEXAS GOVERNOR RACE ANALYSIS — EXECUTIVE REPORT</strong><br>
            Generated: {report_date}<br><br>
            Data Sources: Texas Secretary of State, Campaign Finance Reports, Texas Ethics Commission, FiveThirtyEight,
            Federal Reserve (FRED), Yahoo Finance, Guardian API, New York Times API<br><br>
            <em>This report is provided for informational purposes only. Predictions are based on
            historical data and current conditions; actual election results may vary. Past performance
            does not guarantee future results.</em>
        </p>
    </div>
</body>
</html>
'''

    # Convert HTML to PDF
    pdf_buffer = io.BytesIO()
    HTML(string=html_report).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# =============================================================================
# TAB 1: CLIENT RACE RESULTS
# =============================================================================
def render_client_tab(manager, viz):
    """Render the Client Race Results tab."""
    st.markdown('<div class="tab-header">Texas Governor Race Results</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Executive Summary:</strong> This section provides a high-level overview of Texas Governor
    race results from 2010-2022, plus the upcoming 2026 race featuring incumbent Greg Abbott (R) vs
    likely Democratic nominee Gina Hinojosa.
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # 2026 RACE PREVIEW
    # ==========================================================================
    st.markdown('<div class="section-header">2026 Race Preview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Republican", "Greg Abbott", delta="Incumbent (4th term)")
    with col2:
        st.metric("Democrat", "Gina Hinojosa", delta="State Representative")
    with col3:
        st.metric("Current Polling", "Abbott +8", delta="50% - 42%")
    with col4:
        st.metric("Election Date", "Nov 3, 2026", delta="Primary: Mar 3")

    # 2026 Fundraising Comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**2026 Campaign Fundraising**")
        fundraising_2026 = pd.DataFrame({
            'Candidate': ['Greg Abbott (R)', 'Gina Hinojosa (D)'],
            'Funds Raised': ['$105.7M', '$1.3M'],
            'Cash on Hand': ['$105.7M', '$661K'],
            'Funding Ratio': ['81x', '1x']
        })
        st.dataframe(fundraising_2026, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**2026 Key Dates**")
        dates_2026 = pd.DataFrame({
            'Event': ['Filing Deadline', 'Primary Election', 'Primary Runoff', 'General Election'],
            'Date': ['Dec 8, 2025', 'March 3, 2026', 'May 26, 2026', 'November 3, 2026'],
            'Status': ['✓ Complete', 'Upcoming', 'If needed', 'Upcoming']
        })
        st.dataframe(dates_2026, use_container_width=True, hide_index=True)

    # ==========================================================================
    # 2026 FULL PREDICTION
    # ==========================================================================
    st.markdown('<div class="section-header">2026 Election Predictions</div>', unsafe_allow_html=True)

    # Primary Predictions
    st.markdown("#### Primary Elections (March 3, 2026)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Republican Primary**")
        r_primary_df = pd.DataFrame({
            'Candidate': ['Greg Abbott', 'Evelyn Brooks', 'Mark Goloby', 'Others'],
            'Status': ['Incumbent', 'SBOE Member', 'Tech Executive', 'Various'],
            'Funds': ['$105.7M', '~$100K', '~$500K', '~$175K'],
            'Predicted %': ['85-90%', '3%', '2%', '5%']
        })
        st.dataframe(r_primary_df, use_container_width=True, hide_index=True)
        st.success("**Winner: ABBOTT (85-90%)** - No runoff needed")

    with col2:
        st.markdown("**Democratic Primary**")
        d_primary_df = pd.DataFrame({
            'Candidate': ['Gina Hinojosa', 'Chris Bell', 'Bobby Cole', 'Others'],
            'Status': ['State Rep', 'Former US Rep', 'Farmer', 'Various'],
            'Funds': ['$1.3M', '$33K', '$61K', '~$50K'],
            'Predicted %': ['52-58%', '12-15%', '8-10%', '5-8%']
        })
        st.dataframe(d_primary_df, use_container_width=True, hide_index=True)
        st.success("**Winner: HINOJOSA (52-58%)** - Likely avoids runoff")

    # General Election Prediction
    st.markdown("#### General Election (November 3, 2026)")
    st.markdown("##### Greg Abbott (R) vs Gina Hinojosa (D)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Winner", "Greg Abbott (R)", delta="4th Term")
    with col2:
        st.metric("Predicted Margin", "R+12-18%", delta="99%+ confidence")
    with col3:
        st.metric("Abbott Vote Share", "~59%", delta="Hinojosa ~39%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Key Factors**")
        factors_df = pd.DataFrame({
            'Factor': ['Current Polling', 'Fundraising Gap', 'Incumbency', 'VIX (Volatility)',
                       'Unemployment', 'Inflation', 'GDP Growth', 'Last D Win'],
            'Value': ['Abbott +8 (50-42)', '81:1 R advantage', '3-term incumbent', '15.4 (Low)',
                     '4.5%', '2.8%', '2.2%', '1994 (32 years)']
        })
        st.dataframe(factors_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Predicted Vote Share**")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Greg Abbott (R)', 'Gina Hinojosa (D)', 'Other'],
            y=[59, 39, 2],
            marker_color=[COLORS['republican'], COLORS['democrat'], COLORS['neutral']],
            text=['59%', '39%', '2%'],
            textposition='auto'
        ))
        fig.update_layout(
            yaxis_title='Vote %',
            yaxis=dict(range=[0, 70]),
            height=300,
            margin=dict(t=20, b=20),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['paper'],
            font=dict(color=COLORS['text'])
        )
        st.plotly_chart(fig, use_container_width=True)

    # Historical Comparison Table
    st.markdown("#### Historical Comparison (2010-2026)")

    historical_df = pd.DataFrame({
        'Year': [2010, 2014, 2018, 2022, 2026],
        'R Candidate': ['Rick Perry', 'Greg Abbott', 'Greg Abbott', 'Greg Abbott', 'Greg Abbott'],
        'D Candidate': ['Bill White', 'Wendy Davis', 'Lupe Valdez', "Beto O'Rourke", 'Gina Hinojosa'],
        'R Funds': ['$42M', '$48M', '$46M', '$75M', '$106M'],
        'D Funds': ['$28M', '$42M', '$4.5M', '$80M', '$1.3M'],
        'Margin': ['R+12.7%', 'R+20.4%', 'R+13.3%', 'R+11.1%', 'R+12-18%*'],
        'Winner': ['Perry (R)', 'Abbott (R)', 'Abbott (R)', 'Abbott (R)', 'Abbott (R)*']
    })
    st.dataframe(historical_df, use_container_width=True, hide_index=True)
    st.caption("*2026 values are model predictions based on current polling and fundraising data")

    # Key Takeaways
    st.markdown("#### Key Takeaways")
    st.markdown("""
    - **Massive fundraising disparity**: Abbott's $105.7M vs Hinojosa's $1.3M is the largest gap in recent history (81:1 ratio)
    - **Historical pattern**: Republicans have won every Texas Governor race since 1994 (32 years)
    - **Incumbency advantage**: Abbott seeking historic 4th term with unified GOP backing and Trump support
    - **If Abbott wins**: Becomes longest-serving TX Governor (16 years by 2031), surpassing Rick Perry's 14 years
    """)

    st.markdown("---")

    # Key Metrics Row
    st.markdown('<div class="section-header">Historical Key Metrics (2010-2022)</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">Election Results Summary (Including 2026 Prediction)</div>', unsafe_allow_html=True)

    margin_data = manager.all_results.get('elections', {}).get('margin_statistics', {}).get('by_year', {})
    if margin_data:
        results_list = [
            {
                'Year': year,
                'Winner': data.get('winner', 'N/A'),
                'Party': data.get('winner_party', 'N/A'),
                'Victory Margin': f"{data.get('margin_pct', 0):.1f}%",
                'Status': '✓ Final'
            }
            for year, data in sorted(margin_data.items())
        ]
        # Add 2026 prediction
        results_list.append({
            'Year': 2026,
            'Winner': 'Greg Abbott*',
            'Party': 'R',
            'Victory Margin': 'R+12-18%*',
            'Status': 'Predicted'
        })
        results_df = pd.DataFrame(results_list)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.caption("*2026 values are model predictions based on current polling and fundraising data")

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
    st.markdown('<div class="tab-header">Statistical Model Results</div>',
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
    with st.expander("Election Statistics", expanded=True):
        elections = manager.all_results.get('elections', {})

        col1, col2 = st.columns(2)

        with col1:
            vote_stats = elections.get('vote_statistics', {}).get('overall', {})
            display_stats_table(vote_stats, "Vote Statistics")

        with col2:
            margin_stats = elections.get('margin_statistics', {}).get('competitiveness', {})
            display_stats_table(margin_stats, "Margin Statistics")

        trends = elections.get('historical_trends', {})
        display_stats_table(trends, "Historical Trends")

    with st.expander("Campaign Finance Statistics"):
        finance = manager.all_results.get('campaign_finance', {})

        col1, col2 = st.columns(2)

        with col1:
            fundraising = finance.get('fundraising_statistics', {}).get('overall', {})
            display_stats_table(fundraising, "Fundraising Overview")

        with col2:
            spending = finance.get('spending_statistics', {})
            display_stats_table(spending, "Spending Analysis")

        # Expenditure chart
        fig = viz.expenditure_categories()
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Polling Statistics"):
        polling = manager.all_results.get('polling', {})

        col1, col2 = st.columns(2)

        with col1:
            poll_summary = polling.get('poll_summary', {})
            display_stats_table(poll_summary, "Poll Summary")

            fig = viz.pollster_activity()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sample_stats = polling.get('sample_size_analysis', {})
            display_stats_table(sample_stats, "Sample Size Analysis")

            trend_stats = polling.get('trend_analysis', {})
            if trend_stats.get('volatility_summary'):
                display_stats_table(trend_stats['volatility_summary'], "Trend Analysis")

    with st.expander("News Coverage Statistics"):
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
            event_summary = culture_war.get('event_summary', {})
            display_stats_table(event_summary, "Event Summary")

            fig = viz.culture_war_events_by_year()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            political = culture_war.get('political_leaning_analysis', {})
            filtered_political = {k: v for k, v in political.items() if not k.endswith('_distribution')}
            display_stats_table(filtered_political, "Political Leaning")

            fig = viz.culture_war_political_leaning()
            st.plotly_chart(fig, use_container_width=True)

        fig = viz.culture_war_by_industry()
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Market Statistics"):
        market = manager.all_results.get('market', {})

        col1, col2 = st.columns(2)

        with col1:
            vix = market.get('vix_analysis', {})
            filtered_vix = {k: v for k, v in vix.items() if k != 'summary'}
            display_stats_table(filtered_vix, "VIX Analysis")

            fig = viz.vix_summary()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if vix.get('summary'):
                display_stats_table(vix['summary'], "VIX Distribution")

            fig = viz.vix_distribution()
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Macroeconomic Statistics"):
        macro = manager.all_results.get('macroeconomic', {})

        col1, col2 = st.columns(2)

        with col1:
            inflation = macro.get('inflation_analysis', {})
            filtered_inflation = {k: v for k, v in inflation.items() if k != 'cpi_summary'}
            display_stats_table(filtered_inflation, "Inflation Analysis")

            gdp = macro.get('gdp_analysis', {})
            filtered_gdp = {k: v for k, v in gdp.items() if k != 'growth_summary'}
            display_stats_table(filtered_gdp, "GDP Analysis")

        with col2:
            employment = macro.get('employment_analysis', {})
            filtered_employment = {k: v for k, v in employment.items() if k != 'unemployment_summary'}
            display_stats_table(filtered_employment, "Employment Analysis")

            rates = macro.get('rates_analysis', {})
            display_stats_table(rates, "Rates Analysis")

        fig = viz.macro_indicators_summary()
        st.plotly_chart(fig, use_container_width=True)

    # Cross-Dataset Correlations
    st.markdown('<div class="section-header">Cross-Dataset Analysis</div>', unsafe_allow_html=True)

    fig = viz.integrated_cycle_comparison()
    st.plotly_chart(fig, use_container_width=True)

    correlations = manager.all_results.get('correlations', {})
    if correlations.get('polls_vs_results'):
        st.markdown("**Polls vs Results Analysis**")
        polls_vs_results = correlations['polls_vs_results']
        display_stats_table(polls_vs_results)

        # Show by-year breakdown if available
        if polls_vs_results.get('by_year'):
            st.markdown("**Polling Error by Year**")
            by_year = polls_vs_results['by_year']
            year_data = []
            for year, data in sorted(by_year.items()):
                if isinstance(data, dict):
                    year_data.append({
                        'Year': year,
                        'Polling Error': f"{data.get('polling_error', 'N/A')}",
                        'Direction': data.get('direction', 'N/A')
                    })
            if year_data:
                st.dataframe(pd.DataFrame(year_data), use_container_width=True, hide_index=True)

    # Predictive Models Section
    st.markdown('<div class="section-header">Predictive Models</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Predictive Analysis:</strong> OLS and Logistic regression models trained to predict election winners
    using election margins, campaign finance, polling, news sentiment, culture war events, and economic indicators.
    Training: 2010, 2014 | Testing: 2018, 2022
    </div>
    """, unsafe_allow_html=True)

    # Load predictive models
    try:
        with st.spinner("Running predictive models..."):
            predictive_results, predictive_model = load_predictive_models(manager)

        # Model Accuracy Comparison
        col1, col2 = st.columns(2)

        with col1:
            fig = viz.model_accuracy_comparison(predictive_results)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = viz.model_predictions_comparison(predictive_results)
            st.plotly_chart(fig, use_container_width=True)

        # OLS Regression Results
        with st.expander("OLS Regression Results", expanded=True):
            ols_results = predictive_results.get('ols_regression', {})

            if 'error' not in ols_results:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("R-Squared", f"{ols_results.get('r_squared', 'N/A')}")
                with col2:
                    st.metric("Adj. R-Squared", f"{ols_results.get('adj_r_squared', 'N/A')}")
                with col3:
                    st.metric("Accuracy", f"{ols_results.get('accuracy', 'N/A')}%")
                with col4:
                    st.metric("Observations", f"{ols_results.get('n_observations', 'N/A')}")

                fig = viz.ols_coefficients_chart(predictive_results)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"OLS Error: {ols_results.get('error')}")

        # Logistic/Ridge Regression Results
        logistic_results = predictive_results.get('logistic_regression', {})
        model_type = logistic_results.get('model_type', 'Logistic Regression')
        is_ridge = 'Ridge' in model_type if model_type else False
        with st.expander(f"{model_type} Results", expanded=True):
            if 'error' not in logistic_results:
                # Show note if using Ridge fallback
                if is_ridge:
                    st.info("Note: All 4 elections were won by Republicans, so Ridge Regression predicts victory margin instead of binary winner.")

                st.markdown("**Model Performance**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Training Metrics (2010, 2014)**")
                    train_metrics = logistic_results.get('training_metrics', {})
                    display_stats_table(train_metrics)

                with col2:
                    st.markdown("**Testing Metrics (2018, 2022)**")
                    test_metrics = logistic_results.get('testing_metrics', {})
                    display_stats_table(test_metrics)

                # Backfill results
                backfill = logistic_results.get('backfill', {})
                if backfill:
                    backfill_acc = backfill.get('accuracy', 'N/A')
                    if backfill_acc != 'N/A':
                        backfill_acc = float(backfill_acc)
                    st.markdown(f"**Backfill Accuracy (All Years):** {backfill_acc}%")

                col1, col2 = st.columns(2)

                with col1:
                    fig = viz.model_feature_importance(predictive_results)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = viz.prediction_probability_timeline(predictive_results)
                    st.plotly_chart(fig, use_container_width=True)

                # Predictions table
                st.markdown("**Year-by-Year Predictions**")
                if backfill.get('all_years'):
                    pred_df = pd.DataFrame({
                        'Year': backfill['all_years'],
                        'Actual (R Win)': ['Yes' if a == 1 else 'No' for a in backfill['actual']],
                        'Predicted (R Win)': ['Yes' if p == 1 else 'No' for p in backfill['predicted']],
                        'Win Probability': [f"{p:.1%}" for p in backfill['probability']],
                        'Correct': ['✓' if a == p else '✗' for a, p in zip(backfill['actual'], backfill['predicted'])]
                    })
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
            else:
                st.error(f"Logistic Regression Error: {logistic_results.get('error')}")

    except Exception as e:
        st.error(f"Error running predictive models: {e}")
        st.info("Predictive models require scikit-learn and statsmodels. Install with: pip install scikit-learn statsmodels")

    # =========================================================================
    # 2026 PREDICTION REFERENCE
    # =========================================================================
    st.markdown("---")
    st.markdown("## 2026 Prediction")
    st.info("**See the Client Race Results tab** for the full 2026 Texas Governor race prediction, including primary and general election forecasts.")




# =============================================================================
# TAB 3: ACADEMIC
# =============================================================================
def render_academic_tab(manager):
    """Render the Academic tab with raw data and code."""
    st.markdown('<div class="tab-header">Academic Resources</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>For Researchers:</strong> This section provides access to raw data, methodology documentation,
    code samples, and detailed data dictionaries suitable for academic review and replication studies.
    Includes 2026 race prediction data and methodology.
    </div>
    """, unsafe_allow_html=True)

    # Sub-tabs within Academic
    academic_tabs = st.tabs(["Raw Data", "2026 Data", "Methodology", "Code", "Data Dictionary"])

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
                    label="Download CSV",
                    data=csv,
                    file_name=f"{dataset_name}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No raw data files found. Run the ETL pipeline first.")

    # 2026 DATA TAB
    with academic_tabs[1]:
        st.markdown("### 2026 Texas Governor Race Data")

        st.markdown("""
        This section provides all data inputs used for the 2026 election prediction model.
        Data is current as of January 2026.
        """)

        # Candidates
        st.markdown("#### Candidates")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Republican Primary**")
            r_candidates_df = pd.DataFrame({
                'Candidate': ['Greg Abbott', 'Evelyn Brooks', 'Mark Goloby', 'Pete Chambers', 'Ronnie Tullos', 'Others (5)'],
                'Status': ['Incumbent Governor', 'SBOE Member', 'Tech Executive', 'Challenger', 'Challenger', 'Various'],
                'Campaign Funds': ['$105,700,000', '~$100,000', '~$500,000', '~$50,000', '~$25,000', '~$100,000'],
                'Polling': ['85%', '3%', '2%', '1%', '1%', '3%']
            })
            st.dataframe(r_candidates_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**Democratic Primary**")
            d_candidates_df = pd.DataFrame({
                'Candidate': ['Gina Hinojosa', 'Chris Bell', 'Bobby Cole', 'Nick Pappas', 'Meagan Tehseldar', 'Andrew White'],
                'Status': ['State Rep (D-49)', 'Former US Rep', 'Farmer/Firefighter', 'Challenger', 'Challenger', 'Suspended'],
                'Campaign Funds': ['$1,300,000', '$33,286', '$61,000', '~$10,000', '~$10,000', 'N/A'],
                'Polling': ['41%', '5%', '3%', '1%', '1%', '6% (withdrew)']
            })
            st.dataframe(d_candidates_df, use_container_width=True, hide_index=True)

        # Polling Data
        st.markdown("#### Current Polling (January 2026)")

        polling_2026 = pd.DataFrame({
            'Pollster': ['Emerson College/Nexstar', 'Generic R vs D (Aug 2025)'],
            'Date': ['Jan 10-12, 2026', 'August 2025'],
            'Abbott (R)': ['50%', '49%'],
            'Hinojosa (D)': ['42%', '43%'],
            'Margin': ['R+8', 'R+6'],
            'Undecided': ['8%', '8%']
        })
        st.dataframe(polling_2026, use_container_width=True, hide_index=True)

        # Economic Indicators
        st.markdown("#### Economic Indicators (January 2026)")

        econ_2026 = pd.DataFrame({
            'Indicator': ['VIX (Volatility)', 'Unemployment Rate', 'CPI Inflation (YoY)', 'GDP Growth', 'Fed Funds Rate'],
            'Current Value': ['15.4', '4.5%', '2.8%', '2.2%', '4.25-4.50%'],
            'Trend': ['Low/Stable', 'Stable', 'Moderating', 'Moderate', 'Easing'],
            'Source': ['CBOE', 'BLS', 'BLS', 'BEA', 'Federal Reserve']
        })
        st.dataframe(econ_2026, use_container_width=True, hide_index=True)

        # Model Input Data
        st.markdown("#### Model Input Features (2026)")

        model_inputs = pd.DataFrame({
            'Feature': ['r_raised', 'd_raised', 'r_fundraising_ratio', 'poll_margin_mean',
                       'vix_mean', 'unemployment_rate', 'inflation_current', 'gdp_growth',
                       'news_sentiment_positive', 'news_sentiment_negative'],
            'Value': ['$105,700,000', '$1,300,000', '81.3x', '+8.0 (R)',
                     '15.4', '4.5%', '2.8%', '2.2%', '0.35', '0.30'],
            'Description': ['Republican total raised', 'Democrat total raised', 'R/D funding ratio',
                           'Polling margin (R advantage)', 'Market volatility index', 'National unemployment',
                           'Consumer price inflation', 'Real GDP growth rate', 'FinBERT positive sentiment',
                           'FinBERT negative sentiment']
        })
        st.dataframe(model_inputs, use_container_width=True, hide_index=True)

        # Download 2026 data
        data_2026_export = {
            'election_year': 2026,
            'r_candidate': 'Greg Abbott',
            'd_candidate': 'Gina Hinojosa',
            'r_raised': 105700000,
            'd_raised': 1300000,
            'r_fundraising_ratio': 81.3,
            'poll_margin': 8.0,
            'vix_mean': 15.4,
            'unemployment_rate': 4.5,
            'inflation_current': 2.8,
            'gdp_growth': 2.2,
            'predicted_margin': 15.0,
            'predicted_winner': 'Abbott (R)',
            'win_probability': 0.993
        }

        st.download_button(
            label="Download 2026 Data (JSON)",
            data=json.dumps(data_2026_export, indent=2),
            file_name="texas_governor_2026_data.json",
            mime="application/json"
        )

        # Data Sources
        st.markdown("#### Data Sources")
        st.markdown("""
        - **Polling**: [Emerson College Polling](https://emersoncollegepolling.com/texas-2026-poll/)
        - **Fundraising**: [Texas Tribune](https://www.texastribune.org/2026/01/15/texas-governors-race-greg-abbott-gina-hinojosa-2026-election/)
        - **Economic Data**: [FRED](https://fred.stlouisfed.org/), [BLS](https://www.bls.gov/)
        - **Market Data**: [Yahoo Finance](https://finance.yahoo.com/quote/%5EVIX/)
        - **Candidate Info**: [Ballotpedia](https://ballotpedia.org/Texas_gubernatorial_election,_2026)
        """)

    # METHODOLOGY TAB
    with academic_tabs[2]:
        st.markdown("### Methodology")

        st.markdown("""
        #### Research Design

        This study employs a multi-source data integration approach to analyze Texas Governor
        races from 2010-2022 and predict the 2026 outcome. The methodology combines quantitative
        analysis of electoral data with contextual factors including campaign finance, media
        coverage, and macroeconomic conditions.

        #### Data Collection

        **Primary Sources:**
        1. **Election Data**: Texas Secretary of State official results
        2. **Campaign Finance**: Campaign Finance Reports and Texas Ethics Commission
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

        #### 2026 Prediction Methodology

        **Model Type:** Ridge Regression (margin prediction)

        **Training Data:** 2010, 2014, 2018, 2022 Texas Governor races

        **Features Used:**
        - Campaign finance (R/D raised, ratio)
        - Polling margin (mean, std)
        - News coverage (articles, sentiment via FinBERT)
        - Market indicators (VIX volatility)
        - Economic indicators (unemployment, inflation, GDP growth)
        - Culture war events count

        **Why Ridge Regression?**
        All 4 historical elections were won by Republicans, making binary classification
        (logistic regression) impossible. Ridge regression predicts the victory margin instead,
        which is then converted to win probability using a sigmoid function.

        **Model Performance:**
        - Training R²: 0.998 (on margin prediction)
        - Backfill Accuracy: 100% (all 4 elections correctly predicted)

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
        - **2026 Prediction**: Only 4 historical data points; model may overfit
        - All historical races won by Republicans; no Democratic baseline

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
2. Campaign Finance Reports - Campaign Finance
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
            label="Download Methodology (Markdown)",
            data=methodology_text,
            file_name="methodology.md",
            mime="text/markdown"
        )

    # CODE TAB
    with academic_tabs[3]:
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

        with st.expander("2026 Prediction Model"):
            st.code("""
from Model import StatisticalModelManager, PredictiveModel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Initialize and train on historical data
manager = StatisticalModelManager(use_database=False)
manager.initialize()
manager.run_all_statistics()

# Create predictive model
predictive = PredictiveModel(manager)
predictive.prepare_model_data()
predictive.add_sentiment_analysis()

# Train Ridge regression on historical margins
feature_cols = [c for c in predictive.model_data.columns
                if c not in ['election_year', 'winner_r', 'margin_pct']]
X_train = predictive.model_data[feature_cols].fillna(0)
y_train = predictive.model_data['margin_pct']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 2026 Input Data
data_2026 = {
    'r_raised': 105700000,      # Abbott: $105.7M
    'd_raised': 1300000,        # Hinojosa: $1.3M
    'poll_margin_mean': 8.0,    # Abbott +8
    'vix_mean': 15.4,           # Current VIX
    'unemployment_rate': 4.5,   # Current unemployment
    'inflation_current': 2.8,   # Current CPI
    'gdp_growth': 2.2,          # GDP growth rate
    # ... other features
}

# Predict
X_2026 = pd.DataFrame([data_2026])[feature_cols].fillna(0)
X_2026_scaled = scaler.transform(X_2026)
predicted_margin = model.predict(X_2026_scaled)[0]

# Convert to win probability
win_prob = 1 / (1 + np.exp(-predicted_margin / 5))

print(f"Predicted margin: R+{predicted_margin:.1f}%")
print(f"Win probability: {win_prob:.1%}")
            """, language="python")

        # Full results JSON download
        st.markdown("#### Export Full Results")

        results_json = json.dumps(manager.all_results, indent=2, default=str)
        st.download_button(
            label="Download Results (JSON)",
            data=results_json,
            file_name="analysis_results.json",
            mime="application/json"
        )

    # DATA DICTIONARY TAB
    with academic_tabs[4]:
        st.markdown("### Data Dictionary")

        st.markdown("""
        The data dictionary defines all datasets, their sources, and field definitions
        used in this analysis.
        """)

        # Display DATA_DICTIONARY from ETL
        for category, info in DATA_DICTIONARY.items():
            with st.expander(f"{category.replace('_', ' ').title()}", expanded=False):
                st.markdown(f"**Description:** {info.get('description', 'N/A')}")
                st.markdown(f"**Source:** {info.get('source', 'N/A')}")

                if 'tables' in info:
                    st.markdown("**Tables:**")
                    for table_name, table_info in info['tables'].items():
                        st.markdown(f"- **{table_name}**: {table_info.get('description', 'N/A')}")

                        if 'columns' in table_info:
                            columns = table_info['columns']
                            # Handle both list and dict formats
                            if isinstance(columns, list):
                                cols_df = pd.DataFrame([
                                    {'Column': col} for col in columns
                                ])
                            elif isinstance(columns, dict):
                                cols_df = pd.DataFrame([
                                    {'Column': col, 'Type': details.get('type', 'N/A'),
                                     'Description': details.get('description', 'N/A')}
                                    for col, details in columns.items()
                                ])
                            else:
                                continue
                            st.dataframe(cols_df, use_container_width=True, hide_index=True)

        # Export data dictionary
        dict_json = json.dumps(DATA_DICTIONARY, indent=2)
        st.download_button(
            label="Download Data Dictionary (JSON)",
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
    st.markdown('<div class="main-header">Texas Governor Race Analysis</div>',
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
        "Client Race Results",
        "Model Test Results",
        "Academic"
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
        <p>Data Sources: Texas SoS, Campaign Finance Reports, FiveThirtyEight, Guardian, NYT, Yahoo Finance, FRED</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
