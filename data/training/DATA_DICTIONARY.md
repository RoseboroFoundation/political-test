# Training Dataset Data Dictionary

## Overview
This document describes all fields in the labeled training dataset for the hierarchical election prediction model.

## Target Variable
| Field | Type | Description |
|-------|------|-------------|
| `actual_margin` | float | Final election margin (D - R), positive = Democratic win |
| `dem_winner` | int | Binary indicator (1 = Democrat won, 0 = Republican won) |

## Race Identification
| Field | Type | Description |
|-------|------|-------------|
| `race_id` | string | Unique identifier for each race (STATE_RACETYPE_YEAR) |
| `election_year` | int | Year of election (2010-2024) |
| `state` | string | Two-letter state abbreviation |
| `race_type` | string | Type of race (Governor, Senate, House) |
| `district` | string | Congressional district number (House races only) |

## Candidate Information
| Field | Type | Description |
|-------|------|-------------|
| `dem_candidate` | string | Name of Democratic candidate |
| `rep_candidate` | string | Name of Republican candidate |
| `dem_incumbent` | int | 1 if Democrat is incumbent, 0 otherwise |
| `rep_incumbent` | int | 1 if Republican is incumbent, 0 otherwise |
| `incumbent_running` | int | 1 if any incumbent is running, 0 if open seat |
| `open_seat` | int | 1 if no incumbent running, 0 otherwise |

## State & Political Context
| Field | Type | Description |
|-------|------|-------------|
| `partisan_lean` | float | State partisan lean (R+/D-), from Cook PVI equivalent |
| `previous_margin` | float | Margin from previous election of same race type |
| `cook_rating_numeric` | int | Cook Political Report rating (-2 to +2): -2=Safe D, -1=Lean D, 0=Toss Up, 1=Lean R, 2=Safe R |

## Polling Data (Tier 2 Features)
| Field | Type | Description |
|-------|------|-------------|
| `polling_margin` | float | Final polling average margin (D - R) |
| `polling_n` | int | Number of polls in average |
| `days_to_election` | int | Days before election (for most recent poll) |

## Campaign Finance (Tier 2 Features)
| Field | Type | Description |
|-------|------|-------------|
| `dem_fundraising` | float | Total Democratic candidate fundraising ($) |
| `rep_fundraising` | float | Total Republican candidate fundraising ($) |
| `fundraising_advantage` | float | Log ratio: log(dem_funds/rep_funds), positive = Dem advantage |
| `dem_small_dollar_pct` | float | Percentage of Dem donations from small donors (<$200) |

## National Economic Indicators (Tier 1 Features)
| Field | Type | Description |
|-------|------|-------------|
| `national_unemployment` | float | National unemployment rate (%) at election time |
| `national_gdp_growth` | float | National GDP growth rate (%) for election quarter |
| `national_inflation` | float | National CPI inflation rate (%) |
| `consumer_confidence` | float | Consumer Confidence Index |
| `presidential_approval` | float | Presidential approval rating (%) |

## State Economic Indicators (Tier 2 Features)
| Field | Type | Description |
|-------|------|-------------|
| `state_unemployment` | float | State unemployment rate (%) |
| `state_gdp_growth` | float | State GDP growth rate (%) |

## Political Cycle Indicators
| Field | Type | Description |
|-------|------|-------------|
| `midterm_indicator` | int | 1 if midterm election, 0 if presidential year |
| `incumbent_party_nat` | string | Party controlling presidency (D or R) |

## Feature Tiers

### Tier 1 (Universal - Always Available)
- partisan_lean
- previous_margin
- national economic indicators
- midterm_indicator
- incumbent_party_nat
- incumbent indicators

### Tier 2 (Enhanced - Often Available)
- polling_margin, polling_n
- campaign finance data
- state economic indicators
- cook_rating_numeric

### Tier 3 (Race-Specific - Sometimes Available)
- Candidate-specific features (quality scores, experience)
- Local issues/scandals
- Third-party candidate presence

## Data Sources
- **Election Results**: MIT Election Data + Science Lab, state election offices
- **Polling Data**: FiveThirtyEight polling database
- **Campaign Finance**: FEC FECA database
- **Economic Data**: FRED (Federal Reserve Economic Data), BLS
- **Presidential Approval**: Gallup, FiveThirtyEight
- **Partisan Lean**: Cook Political Report, FiveThirtyEight

## Notes
- All margin values are expressed as (Democratic - Republican)
- Positive margins indicate Democratic advantage
- Missing values are handled via hierarchical model with partial pooling
- Cook rating is centered on Toss Up (0), with negative values favoring Democrats
