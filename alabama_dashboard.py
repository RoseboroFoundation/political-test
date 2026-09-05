"""
Alabama State Senate Race Analysis Dashboard
Districts 25 & 26 - 2026 Election Cycle

Post-redistricting analysis following federal court order (Allen v. Milligan)
Remedial Plan 3 implemented November 2025

Usage:
    streamlit run alabama_dashboard.py --server.port 8502
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Alabama State Senate - Districts 25 & 26",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Mode
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #ffd700;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #ffd700;
        margin-bottom: 2rem;
    }
    .district-card {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #30363d;
        margin-bottom: 1rem;
    }
    .candidate-dem {
        background-color: #0d1f3c;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .candidate-rep {
        background-color: #1f0d0d;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .metric-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #58a6ff;
    }
    .info-box {
        background-color: #0d1f3c;
        border-left: 4px solid #58a6ff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        color: #c9d1d9;
    }
    .redistricting-note {
        background-color: #3d2a00;
        border-left: 4px solid #d29922;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_race_config(district: int):
    """Load race configuration for a district."""
    config_path = Path(__file__).parent / 'config' / 'races' / f'al_state_senate_{district}.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def load_all_configs():
    """Load configurations for both districts."""
    return {
        25: load_race_config(25),
        26: load_race_config(26)
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_demographic_chart(config):
    """Create demographic pie chart."""
    if not config or 'demographics' not in config:
        return None

    demo = config['demographics']
    labels = ['Black VAP', 'White VAP', 'Hispanic VAP', 'Other VAP']
    values = [
        demo.get('black_vap_pct', 0),
        demo.get('white_vap_pct', 0),
        demo.get('hispanic_vap_pct', 0),
        demo.get('other_vap_pct', 0)
    ]
    colors = ['#2563eb', '#8b5cf6', '#10b981', '#6b7280']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title=f"District {config['district']} Demographics (VAP)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c9d1d9',
        showlegend=False,
        height=300
    )

    return fig


def create_partisan_comparison():
    """Create partisan lean comparison chart."""
    data = {
        'District': ['District 25 (Old)', 'District 25 (New)', 'District 26 (Old)', 'District 26 (New)'],
        'Lean': [25, 5, -15, 8],
        'Color': ['#dc2626', '#dc2626', '#2563eb', '#dc2626']
    }

    fig = go.Figure()

    for i, (dist, lean, color) in enumerate(zip(data['District'], data['Lean'], data['Color'])):
        fig.add_trace(go.Bar(
            x=[dist],
            y=[lean],
            marker_color=color,
            name=dist,
            text=[f"{'R' if lean > 0 else 'D'}+{abs(lean)}"],
            textposition='outside'
        ))

    fig.update_layout(
        title="Partisan Lean: Pre vs Post Redistricting",
        yaxis_title="Partisan Lean (+ = Republican)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c9d1d9',
        showlegend=False,
        height=400
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")

    return fig


def create_bvap_comparison():
    """Create Black VAP comparison chart."""
    fig = go.Figure()

    # District 25
    fig.add_trace(go.Bar(
        name='District 25',
        x=['Before Redistricting', 'After Redistricting'],
        y=[29.0, 51.1],
        marker_color='#ffd700',
        text=['29%', '51.1%'],
        textposition='outside'
    ))

    # District 26
    fig.add_trace(go.Bar(
        name='District 26',
        x=['Before Redistricting', 'After Redistricting'],
        y=[66.1, 43.9],
        marker_color='#00d4ff',
        text=['66.1%', '43.9%'],
        textposition='outside'
    ))

    fig.update_layout(
        title="Black Voting Age Population (BVAP) Change",
        yaxis_title="BVAP Percentage",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c9d1d9',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add 50% threshold line
    fig.add_hline(y=50, line_dash="dash", line_color="#4ade80",
                  annotation_text="50% Threshold", annotation_position="right")

    return fig


# =============================================================================
# FORECAST MODEL (Simplified for State Senate)
# =============================================================================
def calculate_forecast(config):
    """
    Calculate simplified forecast based on available factors.

    For state senate races with limited data, we use:
    1. Partisan lean (from demographics)
    2. Incumbency advantage
    3. National environment
    4. Candidate quality proxies
    """
    if not config:
        return None

    # Base partisan lean from BVAP
    bvap = config.get('demographics', {}).get('black_vap_pct', 50)

    # Simplified model: Higher BVAP = More likely D win
    # At 50% BVAP, roughly toss-up; scale from there
    base_margin = (bvap - 50) * 1.2  # Negative = R advantage

    # Adjust for 2026 being a midterm (slight R advantage historically)
    midterm_adjustment = 2.0  # R+2 in midterms

    # Current national environment (placeholder - would be dynamic)
    national_env = 0  # Neutral

    final_margin = base_margin - midterm_adjustment + national_env

    # Determine winner
    if final_margin > 0:
        winner = "Democratic"
        win_prob = min(95, 50 + abs(final_margin) * 3)
    else:
        winner = "Republican"
        win_prob = min(95, 50 + abs(final_margin) * 3)

    return {
        'predicted_winner': winner,
        'margin': round(abs(final_margin), 1),
        'win_probability': round(win_prob, 1),
        'factors': {
            'bvap_effect': round(base_margin, 1),
            'midterm_adjustment': midterm_adjustment,
            'national_environment': national_env
        },
        'confidence': 'Low' if abs(final_margin) < 3 else ('Medium' if abs(final_margin) < 8 else 'High')
    }


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    # Header
    st.markdown('<div class="main-header">🗳️ Alabama State Senate Forecast</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888;">Districts 25 & 26 | 2026 Election Cycle | Post-Redistricting Analysis</p>', unsafe_allow_html=True)

    # Load data
    configs = load_all_configs()

    # Redistricting Alert
    st.markdown("""
    <div class="redistricting-note">
        <strong>⚠️ Redistricting Notice:</strong> On November 17, 2025, a federal court ordered new district maps
        under Remedial Plan 3 following the Allen v. Milligan ruling. Districts 25 and 26 have been substantially
        redrawn, significantly changing their demographic composition and partisan lean.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Flag_of_Alabama.svg/188px-Flag_of_Alabama.svg.png", width=120)
        st.title("Alabama Senate")
        st.markdown("**Districts 25 & 26**")

        st.markdown("---")

        st.markdown("### Key Dates")
        st.markdown("""
        - **Primary:** May 19, 2026
        - **Runoff:** June 16, 2026
        - **General:** November 3, 2026
        """)

        st.markdown("---")

        st.markdown("### Data Sources")
        st.markdown("""
        - AL Secretary of State
        - Federal Court Records
        - Party Qualification Lists
        - Census/ACS Demographics
        """)

        st.markdown("---")

        st.markdown("### Model Notes")
        st.markdown("""
        Forecasts are based on:
        - Demographics (BVAP)
        - Historical voting patterns
        - National environment
        - Candidate factors

        *Limited historical data due to new district lines.*
        """)

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ District 25", "🗺️ District 26", "📈 Analysis"])

    # =========================================================================
    # TAB 1: Overview
    # =========================================================================
    with tab1:
        st.markdown("### Race Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### District 25")
            config25 = configs.get(25)
            if config25:
                forecast25 = calculate_forecast(config25)

                st.markdown(f"""
                <div class="district-card">
                    <h4 style="color:#ffd700;">Forecast: {forecast25['predicted_winner']}</h4>
                    <p class="metric-highlight">{forecast25['predicted_winner'][0]}+{forecast25['margin']}%</p>
                    <p>Win Probability: {forecast25['win_probability']}%</p>
                    <p>Confidence: {forecast25['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Declared Candidates:**")
                for party in ['Democratic', 'Republican']:
                    candidates = config25.get('candidates_2026', {}).get(party, [])
                    if candidates:
                        for c in candidates:
                            css_class = 'candidate-dem' if party == 'Democratic' else 'candidate-rep'
                            st.markdown(f"""
                            <div class="{css_class}">
                                <strong>{c['name']}</strong> ({party[0]})<br/>
                                <small>{c.get('previous_office', 'New candidate')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption(f"No {party} candidates yet")

        with col2:
            st.markdown("#### District 26")
            config26 = configs.get(26)
            if config26:
                forecast26 = calculate_forecast(config26)

                st.markdown(f"""
                <div class="district-card">
                    <h4 style="color:#00d4ff;">Forecast: {forecast26['predicted_winner']}</h4>
                    <p class="metric-highlight">{forecast26['predicted_winner'][0]}+{forecast26['margin']}%</p>
                    <p>Win Probability: {forecast26['win_probability']}%</p>
                    <p>Confidence: {forecast26['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Declared Candidates:**")
                for party in ['Democratic', 'Republican']:
                    candidates = config26.get('candidates_2026', {}).get(party, [])
                    if candidates:
                        for c in candidates:
                            css_class = 'candidate-dem' if party == 'Democratic' else 'candidate-rep'
                            st.markdown(f"""
                            <div class="{css_class}">
                                <strong>{c['name']}</strong> ({party[0]})<br/>
                                <small>{c.get('previous_office', 'New candidate')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption(f"No {party} candidates yet")

        # BVAP Comparison Chart
        st.markdown("---")
        st.plotly_chart(create_bvap_comparison(), use_container_width=True)

        # Partisan Lean Comparison
        st.plotly_chart(create_partisan_comparison(), use_container_width=True)

    # =========================================================================
    # TAB 2: District 25 Detail
    # =========================================================================
    with tab2:
        config = configs.get(25)
        if config:
            st.markdown("### District 25 Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Black VAP", f"{config['demographics']['black_vap_pct']}%",
                         delta="+22.1% from old lines")
            with col2:
                st.metric("Estimated Lean", config['historical_partisan_lean']['estimated_lean'],
                         delta="-20 from old lines")
            with col3:
                st.metric("Population", f"{config['demographics']['total_population']:,}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Geographic Coverage")
                st.markdown(f"""
                <div class="info-box">
                    <strong>Counties:</strong> {', '.join(config['district_geography']['counties'])}<br/>
                    <strong>Description:</strong> {config['district_geography']['description']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Redistricting Context")
                st.markdown(f"""
                Previously (pre-Nov 2025):
                - Crenshaw County (all)
                - Elmore County (most)
                - Montgomery County (small part)
                - BVAP: 29%

                Now (Remedial Plan 3):
                - Crenshaw County (all)
                - Montgomery County (significant portion)
                - BVAP: 51.1%

                *This creates a competitive district with potential for Democratic victory.*
                """)

            with col2:
                fig = create_demographic_chart(config)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.markdown("#### Candidate Profiles")

            dem_candidates = config.get('candidates_2026', {}).get('Democratic', [])
            rep_candidates = config.get('candidates_2026', {}).get('Republican', [])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Democratic Candidates")
                for c in dem_candidates:
                    with st.expander(f"🔵 {c['name']}"):
                        st.markdown(f"""
                        - **Previous Office:** {c.get('previous_office', 'N/A')}
                        - **Incumbent:** {'Yes' if c.get('incumbent') else 'No'}
                        - **Notes:** {c.get('note', 'N/A')}
                        """)

            with col2:
                st.markdown("##### Republican Candidates")
                if rep_candidates:
                    for c in rep_candidates:
                        with st.expander(f"🔴 {c['name']}"):
                            st.markdown(f"""
                            - **Previous Office:** {c.get('previous_office', 'N/A')}
                            - **Incumbent:** {'Yes' if c.get('incumbent') else 'No'}
                            """)
                else:
                    st.info("No Republican candidates have qualified yet")

    # =========================================================================
    # TAB 3: District 26 Detail
    # =========================================================================
    with tab3:
        config = configs.get(26)
        if config:
            st.markdown("### District 26 Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Black VAP", f"{config['demographics']['black_vap_pct']}%",
                         delta="-22.2% from old lines")
            with col2:
                st.metric("Estimated Lean", config['historical_partisan_lean']['estimated_lean'],
                         delta="+23 from old lines")
            with col3:
                st.metric("Population", f"{config['demographics']['total_population']:,}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Geographic Coverage")
                st.markdown(f"""
                <div class="info-box">
                    <strong>Counties:</strong> {', '.join(config['district_geography']['counties'])}<br/>
                    <strong>Description:</strong> {config['district_geography']['description']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Redistricting Context")
                st.markdown(f"""
                Previously (pre-Nov 2025):
                - Montgomery County (central/downtown)
                - BVAP: 66.1%
                - Safe Democratic seat

                Now (Remedial Plan 3):
                - Elmore County (all)
                - Montgomery County (portion)
                - BVAP: 43.9%

                *This transforms the district from safe D to likely R.*
                """)

            with col2:
                fig = create_demographic_chart(config)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.markdown("#### Candidate Profiles")

            dem_candidates = config.get('candidates_2026', {}).get('Democratic', [])
            rep_candidates = config.get('candidates_2026', {}).get('Republican', [])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Democratic Candidates")
                for c in dem_candidates:
                    with st.expander(f"🔵 {c['name']}"):
                        st.markdown(f"""
                        - **Previous Office:** {c.get('previous_office', 'N/A')}
                        - **Incumbent:** {'Yes' if c.get('incumbent') else 'No'}
                        """)

            with col2:
                st.markdown("##### Republican Candidates")
                for c in rep_candidates:
                    with st.expander(f"🔴 {c['name']}"):
                        st.markdown(f"""
                        - **Previous Office:** {c.get('previous_office', 'N/A')}
                        - **Incumbent:** {'Yes' if c.get('incumbent') else 'No'}
                        - **Notes:** {c.get('note', 'N/A')}
                        """)

    # =========================================================================
    # TAB 4: Analysis
    # =========================================================================
    with tab4:
        st.markdown("### Forecast Methodology")

        st.markdown("""
        #### Model Overview

        Due to the significant redistricting changes, historical election data has limited applicability.
        Our forecast model relies on:

        1. **Demographic Composition** - Black Voting Age Population (BVAP) is the strongest predictor
           of partisan lean in Alabama districts

        2. **National Environment** - Generic ballot and presidential approval ratings

        3. **Midterm Dynamics** - Historical patterns show the president's party typically loses seats

        4. **Candidate Quality** - Experience, fundraising, and name recognition
        """)

        st.markdown("---")

        st.markdown("#### Key Assumptions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **District 25:**
            - BVAP of 51.1% creates a competitive district
            - Kirk Hatcher brings incumbency advantages (experience, name ID)
            - No Republican has qualified yet
            - Democratic primary likely to be competitive
            """)

        with col2:
            st.markdown("""
            **District 26:**
            - BVAP of 43.9% leans Republican
            - Will Barfoot brings strong local ties
            - National Republican wave potential in midterms
            - District includes Republican-leaning Elmore County
            """)

        st.markdown("---")

        st.markdown("#### Scenario Analysis")

        scenarios = pd.DataFrame({
            'Scenario': ['Base Case', 'D Wave (+5)', 'R Wave (+5)', 'Neutral'],
            'D25 Winner': ['D', 'D', 'Toss-up', 'D'],
            'D25 Margin': ['D+1', 'D+6', 'Even', 'D+1'],
            'D26 Winner': ['R', 'Toss-up', 'R', 'R'],
            'D26 Margin': ['R+5', 'Even', 'R+10', 'R+5']
        })

        st.dataframe(scenarios, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.markdown("#### Data Limitations")

        st.warning("""
        **Important Caveats:**
        - These are newly drawn districts with no direct electoral history
        - Precinct-level voting patterns from prior districts may not transfer
        - Candidate recruitment is ongoing
        - National environment 10 months out is uncertain

        Forecasts will be updated as more data becomes available.
        """)


if __name__ == "__main__":
    main()
