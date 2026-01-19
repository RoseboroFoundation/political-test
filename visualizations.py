"""
Visualizations Module for Texas Governor Race Data Analysis.

This module provides visualization functions for election, campaign finance,
polling, news, culture war, market, and macroeconomic data.

Usage:
    from visualizations import Visualizer
    viz = Visualizer(model_manager)
    fig = viz.election_margin_trend()
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# COLOR SCHEMES - DARK MODE
# =============================================================================
COLORS = {
    'republican': '#f85149',
    'democrat': '#58a6ff',
    'independent': '#8b949e',
    'primary': '#58a6ff',
    'secondary': '#f0883e',
    'success': '#3fb950',
    'warning': '#d29922',
    'neutral': '#8b949e',
    'light_blue': '#388bfd',
    'light_red': '#ff7b72',
    'background': '#0e1117',
    'paper': '#161b22',
    'grid': '#30363d',
    'text': '#c9d1d9'
}

PARTY_COLORS = {
    'R': COLORS['republican'],
    'D': COLORS['democrat'],
    'Republican': COLORS['republican'],
    'Democrat': COLORS['democrat'],
    'I': COLORS['independent'],
    'Independent': COLORS['independent']
}

# Dark mode layout template for all charts
DARK_LAYOUT = dict(
    paper_bgcolor=COLORS['background'],
    plot_bgcolor=COLORS['paper'],
    font=dict(color=COLORS['text']),
    title_font=dict(color=COLORS['text']),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'])
    ),
    xaxis=dict(
        gridcolor=COLORS['grid'],
        linecolor=COLORS['grid'],
        tickfont=dict(color=COLORS['text']),
        title_font=dict(color=COLORS['text'])
    ),
    yaxis=dict(
        gridcolor=COLORS['grid'],
        linecolor=COLORS['grid'],
        tickfont=dict(color=COLORS['text']),
        title_font=dict(color=COLORS['text'])
    )
)


# =============================================================================
# VISUALIZER CLASS
# =============================================================================
class Visualizer:
    """
    Creates visualizations from statistical model results.
    """

    def __init__(self, model_manager=None, results: Dict[str, Any] = None):
        """
        Initialize visualizer with model manager or pre-computed results.

        Args:
            model_manager: StatisticalModelManager instance
            results: Dictionary of pre-computed results
        """
        self.manager = model_manager
        self.results = results or (model_manager.all_results if model_manager else {})

    def _get_data(self, category: str, subcategory: str = None) -> Any:
        """Helper to safely get data from results."""
        if not self.results:
            return None
        data = self.results.get(category, {})
        if subcategory:
            return data.get(subcategory, {})
        return data

    def _apply_dark_theme(self, fig: go.Figure) -> go.Figure:
        """Apply dark theme to a Plotly figure."""
        fig.update_layout(**DARK_LAYOUT)
        return fig

    # =========================================================================
    # ELECTION VISUALIZATIONS
    # =========================================================================
    def election_margin_trend(self) -> go.Figure:
        """Create election margin trend over time."""
        margin_data = self._get_data('elections', 'margin_statistics')
        if not margin_data or 'by_year' not in margin_data:
            return self._empty_figure("No election margin data available")

        years = []
        margins = []
        winners = []
        parties = []

        for year, data in sorted(margin_data['by_year'].items()):
            years.append(int(year))
            margins.append(data.get('margin_pct', 0))
            winners.append(data.get('winner', 'Unknown'))
            parties.append(data.get('winner_party', 'Unknown'))

        df = pd.DataFrame({
            'Year': years,
            'Margin (%)': margins,
            'Winner': winners,
            'Party': parties
        })

        fig = go.Figure()

        # Add bar chart for margins
        colors = [PARTY_COLORS.get(p, COLORS['neutral']) for p in parties]
        fig.add_trace(go.Bar(
            x=df['Year'],
            y=df['Margin (%)'],
            marker_color=colors,
            text=[f"{w}<br>{m:.1f}%" for w, m in zip(winners, margins)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Winner: %{text}<extra></extra>'
        ))

        fig.update_layout(
            title='Texas Governor Race - Victory Margins by Year',
            xaxis_title='Election Year',
            yaxis_title='Victory Margin (%)',
            xaxis=dict(tickmode='array', tickvals=years),
            showlegend=False,
            height=400
        )

        return self._apply_dark_theme(fig)

    def election_vote_totals(self) -> go.Figure:
        """Create vote totals comparison by year."""
        vote_data = self._get_data('elections', 'vote_statistics')
        if not vote_data or 'by_year' not in vote_data:
            return self._empty_figure("No vote data available")

        years = []
        totals = []
        candidates = []

        for year, data in sorted(vote_data['by_year'].items()):
            years.append(int(year))
            totals.append(data.get('total_votes', 0))
            candidates.append(data.get('candidates', 0))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years,
            y=totals,
            mode='lines+markers',
            name='Total Votes',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=12)
        ))

        fig.update_layout(
            title='Texas Governor Race - Total Votes by Year',
            xaxis_title='Election Year',
            yaxis_title='Total Votes',
            xaxis=dict(tickmode='array', tickvals=years),
            yaxis=dict(tickformat=','),
            height=400
        )

        return self._apply_dark_theme(fig)

    def party_performance_comparison(self) -> go.Figure:
        """Create party performance comparison over time."""
        party_data = self._get_data('elections', 'party_performance')
        if not party_data or 'two_party_trend' not in party_data:
            return self._empty_figure("No party performance data available")

        trend = party_data['two_party_trend']
        years = [d['year'] for d in trend]
        r_pct = [d['r_two_party_pct'] for d in trend]
        d_pct = [d['d_two_party_pct'] for d in trend]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Republican',
            x=years,
            y=r_pct,
            marker_color=COLORS['republican']
        ))

        fig.add_trace(go.Bar(
            name='Democrat',
            x=years,
            y=d_pct,
            marker_color=COLORS['democrat']
        ))

        fig.add_hline(y=50, line_dash="dash", line_color="black",
                      annotation_text="50% Threshold")

        fig.update_layout(
            title='Texas Governor Race - Two-Party Vote Share',
            xaxis_title='Election Year',
            yaxis_title='Vote Share (%)',
            barmode='group',
            xaxis=dict(tickmode='array', tickvals=years),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # CAMPAIGN FINANCE VISUALIZATIONS
    # =========================================================================
    def fundraising_by_year(self) -> go.Figure:
        """Create fundraising by year chart."""
        finance_data = self._get_data('campaign_finance', 'fundraising_statistics')
        if not finance_data or 'by_year' not in finance_data:
            return self._empty_figure("No fundraising data available")

        years = []
        totals = []
        avgs = []

        for year, data in sorted(finance_data['by_year'].items()):
            years.append(int(year))
            totals.append(data.get('total_raised', 0))
            avgs.append(data.get('avg_per_candidate', 0))

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(name='Total Raised', x=years, y=totals,
                   marker_color=COLORS['primary']),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(name='Avg per Candidate', x=years, y=avgs,
                       mode='lines+markers', line=dict(color=COLORS['secondary'], width=3)),
            secondary_y=True
        )

        fig.update_layout(
            title='Campaign Fundraising by Election Cycle',
            xaxis_title='Election Year',
            xaxis=dict(tickmode='array', tickvals=years),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400
        )
        fig.update_yaxes(title_text='Total Raised ($)', secondary_y=False, tickformat='$,.0f')
        fig.update_yaxes(title_text='Avg per Candidate ($)', secondary_y=True, tickformat='$,.0f')

        return self._apply_dark_theme(fig)

    def party_fundraising_comparison(self) -> go.Figure:
        """Create party fundraising comparison."""
        party_data = self._get_data('campaign_finance', 'party_comparison')
        if not party_data or 'by_cycle' not in party_data:
            return self._empty_figure("No party comparison data available")

        cycles = party_data['by_cycle']
        years = [d['year'] for d in cycles]
        r_raised = [d.get('r_raised', 0) for d in cycles]
        d_raised = [d.get('d_raised', 0) for d in cycles]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Republican',
            x=years,
            y=r_raised,
            marker_color=COLORS['republican']
        ))

        fig.add_trace(go.Bar(
            name='Democrat',
            x=years,
            y=d_raised,
            marker_color=COLORS['democrat']
        ))

        fig.update_layout(
            title='Fundraising by Party and Election Cycle',
            xaxis_title='Election Year',
            yaxis_title='Total Raised ($)',
            barmode='group',
            xaxis=dict(tickmode='array', tickvals=years),
            yaxis=dict(tickformat='$,.0f'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400
        )

        return self._apply_dark_theme(fig)

    def expenditure_categories(self) -> go.Figure:
        """Create expenditure by category pie chart."""
        exp_data = self._get_data('campaign_finance', 'expenditure_categories')
        if not exp_data or 'categories' not in exp_data:
            return self._empty_figure("No expenditure data available")

        categories = list(exp_data['categories'].keys())
        amounts = [exp_data['categories'][c]['amount'] for c in categories]

        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=amounts,
            hole=0.4,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>'
        )])

        fig.update_layout(
            title='Campaign Expenditures by Category',
            height=450
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # POLLING VISUALIZATIONS
    # =========================================================================
    def polling_margin_by_year(self) -> go.Figure:
        """Create polling margin distribution by year."""
        margin_data = self._get_data('polling', 'margin_statistics')
        if not margin_data or 'by_year' not in margin_data:
            return self._empty_figure("No polling margin data available")

        years = []
        means = []
        stds = []
        mins = []
        maxs = []

        for year, data in sorted(margin_data['by_year'].items()):
            if data:
                years.append(int(year))
                means.append(data.get('mean', 0))
                stds.append(data.get('std', 0))
                mins.append(data.get('min', 0))
                maxs.append(data.get('max', 0))

        fig = go.Figure()

        # Add range
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=maxs + mins[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Range',
            showlegend=True
        ))

        # Add mean line
        fig.add_trace(go.Scatter(
            x=years,
            y=means,
            mode='lines+markers',
            name='Mean Margin',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=10)
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title='Polling Margins by Election Year',
            xaxis_title='Election Year',
            yaxis_title='R - D Margin (%)',
            xaxis=dict(tickmode='array', tickvals=years),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400
        )

        return self._apply_dark_theme(fig)

    def polling_accuracy(self) -> go.Figure:
        """Create polling accuracy analysis chart."""
        accuracy_data = self._get_data('polling', 'polling_accuracy')
        if not accuracy_data or 'by_year' not in accuracy_data:
            return self._empty_figure("No polling accuracy data available")

        years = []
        errors = []
        directions = []

        for year, data in sorted(accuracy_data['by_year'].items()):
            years.append(int(year))
            errors.append(data.get('polling_error', 0) or 0)
            directions.append(data.get('direction', 'N/A'))

        colors = [COLORS['republican'] if e > 0 else COLORS['democrat'] for e in errors]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=years,
            y=errors,
            marker_color=colors,
            text=[f"{e:+.1f}%" for e in errors],
            textposition='auto'
        ))

        fig.add_hline(y=0, line_color="black", line_width=2)

        fig.update_layout(
            title='Polling Error by Election Year<br><sup>Positive = Underestimated Republican, Negative = Overestimated Republican</sup>',
            xaxis_title='Election Year',
            yaxis_title='Polling Error (%)',
            xaxis=dict(tickmode='array', tickvals=years),
            showlegend=False,
            height=400
        )

        return self._apply_dark_theme(fig)

    def pollster_activity(self) -> go.Figure:
        """Create pollster activity chart."""
        pollster_data = self._get_data('polling', 'pollster_analysis')
        if not pollster_data or 'most_active_pollsters' not in pollster_data:
            return self._empty_figure("No pollster data available")

        pollsters = list(pollster_data['most_active_pollsters'].keys())[:10]
        counts = [pollster_data['most_active_pollsters'][p] for p in pollsters]

        fig = go.Figure(go.Bar(
            x=counts,
            y=pollsters,
            orientation='h',
            marker_color=COLORS['primary']
        ))

        fig.update_layout(
            title='Most Active Pollsters',
            xaxis_title='Number of Polls',
            yaxis_title='Pollster',
            height=400,
            yaxis=dict(autorange='reversed')
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # NEWS VISUALIZATIONS
    # =========================================================================
    def news_coverage_by_year(self) -> go.Figure:
        """Create news coverage by year chart."""
        coverage_data = self._get_data('news', 'coverage_summary')
        if not coverage_data or 'by_year' not in coverage_data:
            return self._empty_figure("No news coverage data available")

        years = []
        articles = []
        sources = []

        for year, data in sorted(coverage_data['by_year'].items()):
            years.append(int(year))
            articles.append(data.get('articles', 0))
            sources.append(data.get('sources', 0))

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(name='Articles', x=years, y=articles,
                   marker_color=COLORS['primary']),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(name='Sources', x=years, y=sources,
                       mode='lines+markers', line=dict(color=COLORS['secondary'], width=3)),
            secondary_y=True
        )

        fig.update_layout(
            title='News Coverage by Election Year',
            xaxis_title='Election Year',
            xaxis=dict(tickmode='array', tickvals=years),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=400
        )
        fig.update_yaxes(title_text='Number of Articles', secondary_y=False)
        fig.update_yaxes(title_text='Number of Sources', secondary_y=True)

        return self._apply_dark_theme(fig)

    def news_by_source(self) -> go.Figure:
        """Create news by source pie chart."""
        source_data = self._get_data('news', 'source_analysis')
        if not source_data or 'source_distribution' not in source_data:
            return self._empty_figure("No news source data available")

        sources = list(source_data['source_distribution'].keys())
        counts = list(source_data['source_distribution'].values())

        fig = go.Figure(data=[go.Pie(
            labels=sources,
            values=counts,
            hole=0.4,
            textinfo='label+percent'
        )])

        fig.update_layout(
            title='News Coverage by Source',
            height=450
        )

        return self._apply_dark_theme(fig)

    def news_by_topic(self) -> go.Figure:
        """Create news by topic bar chart."""
        topic_data = self._get_data('news', 'topic_analysis')
        if not topic_data or 'top_topics' not in topic_data:
            return self._empty_figure("No news topic data available")

        topics = list(topic_data['top_topics'].keys())
        counts = list(topic_data['top_topics'].values())

        fig = go.Figure(go.Bar(
            x=counts,
            y=topics,
            orientation='h',
            marker_color=COLORS['primary']
        ))

        fig.update_layout(
            title='Top News Topics',
            xaxis_title='Number of Articles',
            yaxis_title='Topic',
            height=400,
            yaxis=dict(autorange='reversed')
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # CULTURE WAR VISUALIZATIONS
    # =========================================================================
    def culture_war_events_by_year(self) -> go.Figure:
        """Create culture war events by year chart."""
        temporal_data = self._get_data('culture_war', 'temporal_analysis')
        if not temporal_data or 'yearly_trend' not in temporal_data:
            return self._empty_figure("No culture war temporal data available")

        yearly = temporal_data['yearly_trend']
        years = sorted(yearly.keys())
        counts = [yearly[y] for y in years]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8)
        ))

        # Highlight peak year
        peak_year = temporal_data.get('peak_year')
        if peak_year:
            peak_count = yearly.get(peak_year, 0)
            fig.add_annotation(
                x=peak_year, y=peak_count,
                text=f"Peak: {peak_year}",
                showarrow=True, arrowhead=2
            )

        fig.update_layout(
            title='Culture War Events by Year',
            xaxis_title='Year',
            yaxis_title='Number of Events',
            height=400
        )

        return self._apply_dark_theme(fig)

    def culture_war_by_industry(self) -> go.Figure:
        """Create culture war events by industry chart."""
        industry_data = self._get_data('culture_war', 'industry_analysis')
        if not industry_data or 'top_industries' not in industry_data:
            return self._empty_figure("No culture war industry data available")

        industries = list(industry_data['top_industries'].keys())[:10]
        counts = [industry_data['top_industries'][i] for i in industries]

        fig = go.Figure(go.Bar(
            x=counts,
            y=industries,
            orientation='h',
            marker_color=COLORS['secondary']
        ))

        fig.update_layout(
            title='Culture War Events by Industry',
            xaxis_title='Number of Events',
            yaxis_title='Industry',
            height=450,
            yaxis=dict(autorange='reversed')
        )

        return self._apply_dark_theme(fig)

    def culture_war_political_leaning(self) -> go.Figure:
        """Create culture war political leaning pie chart."""
        leaning_data = self._get_data('culture_war', 'political_leaning_analysis')
        if not leaning_data or 'leaning_distribution' not in leaning_data:
            return self._empty_figure("No political leaning data available")

        leanings = list(leaning_data['leaning_distribution'].keys())
        counts = list(leaning_data['leaning_distribution'].values())

        color_map = {
            'Liberal': COLORS['democrat'],
            'Conservative': COLORS['republican'],
            'Mixed': COLORS['neutral']
        }
        colors = [color_map.get(l, COLORS['neutral']) for l in leanings]

        fig = go.Figure(data=[go.Pie(
            labels=leanings,
            values=counts,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent'
        )])

        fig.update_layout(
            title='Culture War Events by Political Leaning',
            height=400
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # MARKET VISUALIZATIONS
    # =========================================================================
    def vix_summary(self) -> go.Figure:
        """Create VIX summary visualization."""
        vix_data = self._get_data('market', 'vix_analysis')
        if not vix_data or 'summary' not in vix_data:
            return self._empty_figure("No VIX data available")

        summary = vix_data['summary']

        # Create a gauge chart for current VIX
        current = vix_data.get('current', 0)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current VIX"},
            delta={'reference': summary.get('mean', 20)},
            gauge={
                'axis': {'range': [0, 80]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 15], 'color': 'lightgreen'},
                    {'range': [15, 25], 'color': 'yellow'},
                    {'range': [25, 40], 'color': 'orange'},
                    {'range': [40, 80], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': summary.get('mean', 20)
                }
            }
        ))

        fig.update_layout(
            title='Market Volatility (VIX) Gauge',
            height=350
        )

        return self._apply_dark_theme(fig)

    def vix_distribution(self) -> go.Figure:
        """Create VIX distribution chart."""
        vix_data = self._get_data('market', 'vix_analysis')
        if not vix_data or 'summary' not in vix_data:
            return self._empty_figure("No VIX data available")

        summary = vix_data['summary']

        # Create box plot style visualization
        fig = go.Figure()

        fig.add_trace(go.Box(
            y=[summary.get('min', 0), summary.get('q25', 0), summary.get('median', 0),
               summary.get('q75', 0), summary.get('max', 0)],
            name='VIX Distribution',
            boxpoints=False,
            marker_color=COLORS['primary']
        ))

        fig.add_hline(y=20, line_dash="dash", line_color="orange",
                      annotation_text="Normal Threshold (20)")
        fig.add_hline(y=30, line_dash="dash", line_color="red",
                      annotation_text="High Volatility (30)")

        fig.update_layout(
            title='VIX Distribution Summary',
            yaxis_title='VIX Value',
            height=400
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # MACROECONOMIC VISUALIZATIONS
    # =========================================================================
    def macro_indicators_summary(self) -> go.Figure:
        """Create macroeconomic indicators summary."""
        macro_data = self._get_data('macroeconomic')
        if not macro_data:
            return self._empty_figure("No macroeconomic data available")

        indicators = []
        values = []
        colors = []

        # Inflation
        inflation = macro_data.get('inflation_analysis', {})
        if inflation.get('current'):
            indicators.append('CPI (YoY)')
            values.append(inflation['current'])
            colors.append(COLORS['warning'] if inflation['current'] > 3 else COLORS['success'])

        # GDP
        gdp = macro_data.get('gdp_analysis', {})
        if gdp.get('current_growth'):
            indicators.append('GDP Growth')
            values.append(gdp['current_growth'])
            colors.append(COLORS['success'] if gdp['current_growth'] > 0 else COLORS['warning'])

        # Unemployment
        employment = macro_data.get('employment_analysis', {})
        if employment.get('current_rate'):
            indicators.append('Unemployment')
            values.append(employment['current_rate'])
            colors.append(COLORS['success'] if employment['current_rate'] < 5 else COLORS['warning'])

        # 10Y Yield
        rates = macro_data.get('rates_analysis', {})
        if rates.get('yield_10y', {}).get('current'):
            indicators.append('10Y Treasury')
            values.append(rates['yield_10y']['current'])
            colors.append(COLORS['primary'])

        if not indicators:
            return self._empty_figure("No macroeconomic indicators available")

        fig = go.Figure(go.Bar(
            x=indicators,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        ))

        fig.update_layout(
            title='Current Macroeconomic Indicators',
            yaxis_title='Value (%)',
            height=400
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # CORRELATION / COMBINED VISUALIZATIONS
    # =========================================================================
    def integrated_cycle_comparison(self) -> go.Figure:
        """Create integrated comparison across election cycles."""
        corr_data = self._get_data('correlations', 'integrated_summary')
        if not corr_data or 'by_cycle' not in corr_data:
            return self._empty_figure("No integrated data available")

        cycles = corr_data['by_cycle']
        years = sorted(cycles.keys())

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Victory Margin', 'Total Fundraising', 'Poll Count', 'News Articles'],
            vertical_spacing=0.15
        )

        margins = []
        fundraising = []
        polls = []
        articles = []

        for year in years:
            cycle = cycles[year]
            margins.append(cycle.get('election', {}).get('margin_pct', 0) or 0)
            fundraising.append(cycle.get('finance', {}).get('total_raised', 0) or 0)
            polls.append(0)  # Placeholder
            articles.append(cycle.get('news', {}).get('articles', 0) or 0)

        # Margins
        fig.add_trace(
            go.Bar(x=years, y=margins, marker_color=COLORS['republican']),
            row=1, col=1
        )

        # Fundraising
        fig.add_trace(
            go.Bar(x=years, y=fundraising, marker_color=COLORS['primary']),
            row=1, col=2
        )

        # Polls (placeholder)
        fig.add_trace(
            go.Bar(x=years, y=polls, marker_color=COLORS['secondary']),
            row=2, col=1
        )

        # Articles
        fig.add_trace(
            go.Bar(x=years, y=articles, marker_color=COLORS['success']),
            row=2, col=2
        )

        fig.update_layout(
            title='Integrated Election Cycle Comparison',
            showlegend=False,
            height=600
        )

        return self._apply_dark_theme(fig)

    # =========================================================================
    # KEY METRICS CARDS DATA
    # =========================================================================
    def get_key_metrics(self) -> Dict[str, Any]:
        """Get key metrics for display cards."""
        metrics = {}

        # Election metrics
        elections = self._get_data('elections')
        if elections:
            margin = elections.get('margin_statistics', {}).get('competitiveness', {})
            metrics['avg_margin'] = margin.get('avg_margin', 'N/A')
            metrics['closest_race'] = margin.get('closest_race', {}).get('year', 'N/A')

        # Finance metrics
        finance = self._get_data('campaign_finance')
        if finance:
            fundraising = finance.get('fundraising_statistics', {}).get('overall', {})
            metrics['avg_raised'] = fundraising.get('mean', 'N/A')
            metrics['money_win_rate'] = finance.get('money_vs_results', {}).get('money_win_rate', 'N/A')

        # Polling metrics
        polling = self._get_data('polling')
        if polling:
            metrics['total_polls'] = polling.get('poll_summary', {}).get('total_polls', 'N/A')
            metrics['polling_error'] = polling.get('polling_accuracy', {}).get('overall', {}).get('mean_absolute_error', 'N/A')

        # News metrics
        news = self._get_data('news')
        if news:
            metrics['total_articles'] = news.get('coverage_summary', {}).get('total_articles', 'N/A')

        # Culture war metrics
        culture_war = self._get_data('culture_war')
        if culture_war:
            metrics['total_events'] = culture_war.get('event_summary', {}).get('total_events', 'N/A')
            metrics['unique_companies'] = culture_war.get('event_summary', {}).get('unique_companies', 'N/A')

        # Market metrics
        market = self._get_data('market')
        if market:
            vix = market.get('vix_analysis', {})
            metrics['vix_current'] = vix.get('current', 'N/A')
            metrics['vix_regime'] = vix.get('regime', 'N/A')

        # Macro metrics
        macro = self._get_data('macroeconomic')
        if macro:
            metrics['current_cpi'] = macro.get('inflation_analysis', {}).get('current', 'N/A')
            metrics['unemployment'] = macro.get('employment_analysis', {}).get('current_rate', 'N/A')

        return metrics

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['text'])
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=300
        )
        return self._apply_dark_theme(fig)


# =============================================================================
# STANDALONE FUNCTIONS FOR QUICK PLOTS
# =============================================================================
def create_metric_card_html(title: str, value: Any, delta: Any = None, color: str = None) -> str:
    """Create HTML for a metric card."""
    color = color or COLORS['primary']
    delta_html = ""
    if delta is not None:
        delta_color = COLORS['success'] if delta >= 0 else COLORS['warning']
        delta_html = f'<span style="color: {delta_color}; font-size: 14px;">{"+" if delta >= 0 else ""}{delta}</span>'

    return f"""
    <div style="
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <p style="color: gray; margin: 0; font-size: 14px;">{title}</p>
        <h2 style="color: {color}; margin: 10px 0;">{value}</h2>
        {delta_html}
    </div>
    """


if __name__ == "__main__":
    # Test visualizations
    print("Visualizations module loaded successfully")
    print("Available classes: Visualizer")
    print("Run with StatisticalModelManager to generate charts")
