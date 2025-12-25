import pandas as pd
import io
import os
import pandas_datareader as pdr
import yfinance as yf
from datetime import datetime
from fredapi import Fred 
from dotenv import load_dotenv



# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Culture War Companies Data Cleaning ---
## Import Culture war companies data from Culture_War_Companies_160_fullmeta.csv
def import_culture_war_data(file_path):
    """
    Imports and cleans the Culture War Companies dataset.
    """
    # Convert relative path to absolute path based on script location
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    df = pd.read_csv(file_path)
    
    ## make "Event Date" a datetime object
    df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
    
    return df

if __name__ == "__main__":
    # Call the function
    df = import_culture_war_data('Culture_War_Companies_160_fullmeta.csv')
    print(df)
    print(f"\nDataframe shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")

#--- END CULTURE WAR COMPANIES DATA CLEANING ---

#---Stock Data for Culture War Companies---
def get_stock_data(tickers, start_date='2000-01-01', end_date='2025-12-31'):
    """
    Downloads stock data for given tickers from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : list
        List of stock tickers
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns:
    --------
    dict
        Dictionary with tickers as keys and dataframes as values
    """
    stock_data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            # Set auto_adjust=False to keep original column structure
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            if not data.empty:
                # Reset index to make Date a column
                data = data.reset_index()
                
                # Add Ticker column at the beginning
                data.insert(0, 'Ticker', ticker)
                
                # The columns should now be: Date, Open, High, Low, Close, Adj Close, Volume
                # Reorder to: Ticker, Date, Open, High, Low, Close, Volume, Adj Close
                column_order = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                data = data[column_order]
                
                stock_data[ticker] = data
                print(f"  ✓ Successfully downloaded {len(data)} rows for {ticker}")
            else:
                failed_tickers.append(ticker)
                print(f"  ✗ No data found for {ticker}")
        
        except Exception as e:
            failed_tickers.append(ticker)
            print(f"  ✗ Error downloading {ticker}: {e}")
    
    if failed_tickers:
        print(f"\nFailed to download data for: {failed_tickers}")
    
    return stock_data

if __name__ == "__main__":
    # Import culture war companies
    df = import_culture_war_data('Culture_War_Companies_160_fullmeta.csv')
    print("Culture War Companies Data:")
    print(df.head())
    print(f"\nDataframe shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    
    # Get unique tickers (adjust column name if needed)
    # Common column names: 'Ticker', 'ticker', 'Symbol', 'Stock Ticker'
    ticker_column = 'Ticker'  # Change this to match your CSV column name
    tickers = df[ticker_column].unique().tolist()
    
    print(f"\nFound {len(tickers)} unique tickers")
    print(f"Tickers: {tickers[:10]}...")  # Show first 10
    
    # Download stock data
    print("\n" + "="*50)
    print("Downloading stock data from 2000-2025...")
    print("="*50 + "\n")
    
    stock_data = get_stock_data(tickers, start_date='2000-01-01', end_date='2025-12-31')
    
    print(f"\n{'='*50}")
    print(f"Successfully downloaded data for {len(stock_data)} out of {len(tickers)} tickers")
    print(f"{'='*50}")
    
    # Example: View data for first ticker
    if stock_data:
        first_ticker = list(stock_data.keys())[0]
        print(f"\nSample data for {first_ticker}:")
        print(stock_data[first_ticker].head())
#--- END Stock Data for Culture War Companies ---

#--VIX DATA----

# ============ CONFIGURATION ============
# Load environment variables FIRST
load_dotenv()

# Get the API key
API_KEY = os.getenv('FRED_API_KEY')

# Date range
START_DATE = '2000-01-01'
END_DATE = '2025-12-31'

# Output file name
OUTPUT_FILE = 'vix_data_2000_2025.csv'
# =======================================

def download_vix_data():
    """Download VIX data from FRED and save to CSV"""
    
    # Check if API key is loaded
    if not API_KEY:
        print("\n❌ ERROR: FRED API key not found or not set!")
        print("\nPlease follow these steps:")
        print("1. Create a file named '.env' in the same directory as this script")
        print("2. Add this line to the .env file:")
        print("   FRED_API_KEY=your_actual_api_key_here")
        print("\n3. Get your free API key at:")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    try:
        # Initialize FRED API
        print("Connecting to FRED...")
        fred = Fred(api_key=API_KEY)
        
        # Download VIX data
        print(f"Downloading VIX data from {START_DATE} to {END_DATE}...")
        vix_series = fred.get_series(
            'VIXCLS',
            observation_start=START_DATE,
            observation_end=END_DATE
        )
        
        # Convert to DataFrame
        vix_df = pd.DataFrame({
            'date': vix_series.index,
            'vix': vix_series.values
        })
        
        # Remove any NaN values
        vix_df = vix_df.dropna()
        
        # Save to CSV
        vix_df.to_csv(OUTPUT_FILE, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"File saved: {OUTPUT_FILE}")
        print(f"Date range: {vix_df['date'].min().date()} to {vix_df['date'].max().date()}")
        print(f"Total observations: {len(vix_df):,}")
        
        print("\n📊 VIX Summary Statistics:")
        print("-"*60)
        stats = vix_df['vix'].describe()
        print(f"Count:  {stats['count']:,.0f}")
        print(f"Mean:   {stats['mean']:.2f}")
        print(f"Std:    {stats['std']:.2f}")
        print(f"Min:    {stats['min']:.2f}")
        print(f"25%:    {stats['25%']:.2f}")
        print(f"50%:    {stats['50%']:.2f}")
        print(f"75%:    {stats['75%']:.2f}")
        print(f"Max:    {stats['max']:.2f}")
        
        # Find extremes
        max_idx = vix_df['vix'].idxmax()
        min_idx = vix_df['vix'].idxmin()
        
        print("\n🔴 Highest VIX:")
        print(f"   {vix_df.loc[max_idx, 'vix']:.2f} on {vix_df.loc[max_idx, 'date'].date()}")
        
        print("🟢 Lowest VIX:")
        print(f"   {vix_df.loc[min_idx, 'vix']:.2f} on {vix_df.loc[min_idx, 'date'].date()}")
        
        print("\nSample Data:")
        print("-"*60)
        print("First 5 observations:")
        print(vix_df.head().to_string(index=False))
        print("\nLast 5 observations:")
        print(vix_df.tail().to_string(index=False))
        
        # Additional analysis
        print("\n📈 Quick Analysis:")
        print(f"Days with VIX > 30: {(vix_df['vix'] > 30).sum():,} ({(vix_df['vix'] > 30).sum()/len(vix_df)*100:.1f}%)")
        print(f"Days with VIX > 40: {(vix_df['vix'] > 40).sum():,} ({(vix_df['vix'] > 40).sum()/len(vix_df)*100:.1f}%)")
        print(f"Days with VIX > 50: {(vix_df['vix'] > 50).sum():,} ({(vix_df['vix'] > 50).sum()/len(vix_df)*100:.1f}%)")
        
        print("="*60)
        
        print("\n✨ Success! Your VIX data is ready to use.")
        print("\n📄 Citation:")
        print("Chicago Board Options Exchange, CBOE Volatility Index: VIX [VIXCLS],")
        print("retrieved from FRED, Federal Reserve Bank of St. Louis;")
        print("https://fred.stlouisfed.org/series/VIXCLS")
        
        return vix_df
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify your API key is correct in the .env file")
        print("2. Check your internet connection")
        return None

if __name__ == "__main__":
    print("="*60)
    print("VIX DATA DOWNLOADER - Using .env Configuration")
    print("="*60)
    print()
    
    vix_df = download_vix_data()

#---FAMA FRENCH DATA----
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import os

def download_fama_french_factors(
    start_date='1926-07-01',
    end_date=None,
    frequency='daily',
    save_path=None
):
    """
    Download Fama-French factor data from Kenneth French's data library.
    
    Parameters:
    -----------
    start_date : str, default '1926-07-01'
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format (defaults to today)
    frequency : str, default 'daily'
        'daily', 'monthly', or 'annual'
    save_path : str, optional
        Path to save CSV file
    
    Returns:
    --------
    dict : Dictionary containing DataFrames for different factor models
    """
    
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Map frequency to dataset names
    freq_map = {
        'daily': 'F-F_Research_Data_Factors_daily',
        'monthly': 'F-F_Research_Data_Factors',
        'annual': 'F-F_Research_Data_Factors'
    }
    
    results = {}
    
    try:
        # Download 3-Factor Model (Mkt-RF, SMB, HML, RF)
        print("Downloading Fama-French 3-Factor Model...")
        ff3 = pdr.DataReader(
            freq_map[frequency],
            'famafrench',
            start=start_date,
            end=end_date
        )[0]  # [0] gets the main dataset
        
        results['FF3'] = ff3
        
        # Download 5-Factor Model (adds RMW, CMA)
        print("Downloading Fama-French 5-Factor Model...")
        ff5_name = 'F-F_Research_Data_5_Factors_2x3_daily' if frequency == 'daily' else 'F-F_Research_Data_5_Factors_2x3'
        ff5 = pdr.DataReader(
            ff5_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        
        results['FF5'] = ff5
        
        # Download Momentum Factor
        print("Downloading Momentum Factor...")
        mom_name = 'F-F_Momentum_Factor_daily' if frequency == 'daily' else 'F-F_Momentum_Factor'
        mom = pdr.DataReader(
            mom_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        
        results['MOM'] = mom
        
        # Save to CSV if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            for name, df in results.items():
                filename = f"{name}_{frequency}_{start_date}_to_{end_date}.csv"
                filepath = os.path.join(save_path, filename)
                df.to_csv(filepath)
                print(f"Saved {name} to {filepath}")
        
        print(f"\nDownload complete! Date range: {ff3.index[0]} to {ff3.index[-1]}")
        return results
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def download_industry_portfolios(
    num_industries=10,
    start_date='1926-07-01',
    end_date=None,
    frequency='daily',
    save_path=None
):
    """
    Download Fama-French industry portfolio returns.
    
    Parameters:
    -----------
    num_industries : int
        Number of industries (5, 10, 12, 17, 30, 38, 48, or 49)
    """
    
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    freq_suffix = '_daily' if frequency == 'daily' else ''
    dataset_name = f'{num_industries}_Industry_Portfolios{freq_suffix}'
    
    try:
        print(f"Downloading {num_industries} Industry Portfolios...")
        ind_portfolios = pdr.DataReader(
            dataset_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"Industry_{num_industries}_{frequency}_{start_date}_to_{end_date}.csv"
            filepath = os.path.join(save_path, filename)
            ind_portfolios.to_csv(filepath)
            print(f"Saved to {filepath}")
        
        return ind_portfolios
        
    except Exception as e:
        print(f"Error downloading industry portfolios: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Download daily data from 2000 onwards
    data = download_fama_french_factors(
        start_date='2000-01-01',
        frequency='daily',
        save_path='./fama_french_data'
    )
    
    # Access specific factors
    if data:
        ff3 = data['FF3']
        ff5 = data['FF5']
        
        print("\nFF3 Factor Sample:")
        print(ff3.head())
        print(f"\nFF3 Shape: {ff3.shape}")
        print(f"Columns: {ff3.columns.tolist()}")
        
        # Download industry portfolios
industries = download_industry_portfolios(
    num_industries=10,
    start_date='2000-01-01',
    frequency='daily',
    save_path='./fama_french_data'
)
        
#-- SEC FORM 4 DATA--
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import os
from typing import List, Dict
import json

class Form4Downloader:
    """Download and parse SEC Form 4 filings"""
    
    def __init__(self, output_dir='./sec_form4_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.headers = {
            'User-Agent': 'Ashley Roseboro ashley@roseboroholdings.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
    
    def get_company_cik(self, ticker: str) -> str:
        # ... your existing code ...
        pass
    
    def download_form4_filings(self, ticker: str, cik: str, start_date: str = '2000-01-01', end_date: str = '2025-12-31') -> List[Dict]:
        # ... your existing code ...
        pass
    
    def parse_form4_xml(self, filing_url: str) -> List[Dict]:
        # ... your existing code ...
        pass
    
    def build_form4_dataset(self, culture_war_companies: List[str], start_date: str = '2000-01-01', end_date: str = '2025-12-31', save_csv: bool = True) -> pd.DataFrame:
        """Build complete Form 4 dataset"""
        all_transactions = []
        
        for ticker in culture_war_companies:
            print(f"\nProcessing {ticker}...")
            
            cik = self.get_company_cik(ticker)
            if not cik:
                continue
            
            filings = self.download_form4_filings(ticker, cik, start_date, end_date)
            
            for filing in filings:
                transactions = self.parse_form4_xml(filing['filing_url'])
                
                for trans in transactions:
                    trans['ticker'] = ticker
                    trans['cik'] = cik
                    trans['filing_date'] = filing['filing_date']
                    trans['accession_number'] = filing.get('accession_number', '')
                    trans['filing_url'] = filing['filing_url']
                    
                all_transactions.extend(transactions)
                time.sleep(0.15)
        
        df = pd.DataFrame(all_transactions)
        
        if len(df) > 0:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
            df['transaction_value'] = df['shares'] * df['price_per_share']
            df = df.sort_values('transaction_date')
            
            if save_csv:
                output_file = os.path.join(self.output_dir, f'form4_transactions_{start_date}_to_{end_date}.csv')
                df.to_csv(output_file, index=False)
                print(f"\n✓ Saved {len(df)} transactions to {output_file}")
        
        return df
    

def download_form4_filings(
    self,
    ticker: str,
    cik: str,
    start_date: str = '2000-01-01',
    end_date: str = '2025-12-31'
) -> List[Dict]:
    """
    Download all Form 4 filings for a company within date range using RSS feed.
    
    Returns:
    --------
    List[Dict] : List of filing metadata
    """
    filings = []
    
    try:
        # Use EDGAR full-text search API
        # Strip leading zeros from CIK for the URL
        cik_no_leading = str(int(cik))
        
        # Build URL for company filings
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_no_leading}&type=4&dateb=&owner=include&count=100&search_text="
        
        response = requests.get(url, headers=self.headers)
        time.sleep(0.2)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all filing rows
            filing_table = soup.find('table', {'class': 'tableFile2'})
            
            if not filing_table:
                print(f"  No Form 4 filings table found for {ticker}")
                return []
            
            rows = filing_table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    filing_type = cols[0].text.strip()
                    
                    if filing_type == '4':
                        filing_date = cols[3].text.strip()
                        
                        # Check date range
                        if start_date <= filing_date <= end_date:
                            # Get document link
                            doc_link = cols[1].find('a')
                            if doc_link:
                                doc_url = 'https://www.sec.gov' + doc_link.get('href')
                                accession = cols[4].text.strip()
                                
                                filing_info = {
                                    'ticker': ticker,
                                    'cik': cik,
                                    'filing_date': filing_date,
                                    'accession_number': accession,
                                    'filing_url': doc_url
                                }
                                filings.append(filing_info)
            
            if len(filings) > 0:
                print(f"  ✓ Found {len(filings)} Form 4 filings for {ticker}")
            else:
                print(f"  No Form 4 filings in date range for {ticker}")
            
            return filings
        else:
            print(f"  HTTP {response.status_code} for {ticker}")
            return []
            
    except Exception as e:
        print(f"  Error downloading Form 4 filings for {ticker}: {e}")
        return []
    
    def parse_form4_xml(self, filing_url: str) -> List[Dict]:
        """
        Parse Form 4 to extract insider trading details.
        First fetches the filing page, then finds and parses the XML document.
        
        Returns:
        --------
        List[Dict] : Parsed transaction data
        """
        try:
            # First, get the filing page to find the actual XML document
            response = requests.get(filing_url, headers=self.headers)
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching filing page: {e}")
            return []
        
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the XML document link
    xml_link = None
    for link in soup.find_all('a'):
        href = link.get('href', '')
        if '.xml' in href.lower() and 'primary_doc' not in href:
            xml_link = 'https://www.sec.gov' + href if not href.startswith('http') else href
            break
    
    if not xml_link:
        print(f"    No XML document found")
        return []
    
    # Now fetch and parse the XML
    xml_response = requests.get(xml_link, headers=self.headers)
    time.sleep(0.2)
    
    if xml_response.status_code != 200:
        return []
    
    xml_soup = BeautifulSoup(xml_response.content, 'xml')
    
    # Extract reporting owner
    reporting_owner = xml_soup.find('reportingOwner')
    if reporting_owner:
        owner_name_tag = reporting_owner.find('rptOwnerName')
        owner_name = owner_name_tag.text if owner_name_tag else 'Unknown'
    else:
        owner_name = 'Unknown'
    
    transactions = []
    
    # Parse non-derivative transactions
    for trans in xml_soup.find_all('nonDerivativeTransaction'):
        try:
            trans_date_tag = trans.find('transactionDate')
            trans_code_tag = trans.find('transactionCode')
            shares_tag = trans.find('transactionShares')
            price_tag = trans.find('transactionPricePerShare')
            acq_disp_tag = trans.find('transactionAcquiredDisposedCode')
            shares_owned_tag = trans.find('sharesOwnedFollowingTransaction')
            
            transaction = {
                'owner_name': owner_name,
                'transaction_date': trans_date_tag.find('value').text if trans_date_tag and trans_date_tag.find('value') else None,
                'transaction_code': trans_code_tag.text if trans_code_tag else None,
                'shares': float(shares_tag.find('value').text) if shares_tag and shares_tag.find('value') else 0,
                'price_per_share': float(price_tag.find('value').text) if price_tag and price_tag.find('value') else None,
                'acquired_disposed': acq_disp_tag.find('value').text if acq_disp_tag and acq_disp_tag.find('value') else None,
                'shares_owned_after': float(shares_owned_tag.find('value').text) if shares_owned_tag and shares_owned_tag.find('value') else None
            }
            transactions.append(transaction)
        except Exception as e:
            continue
    
    return transactions
    
    def build_form4_dataset(
        self,
        culture_war_companies: List[str],
        start_date: str = '2000-01-01',
        end_date: str = '2025-12-31',
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Build complete Form 4 dataset for all culture war companies.
        
        Parameters:
        -----------
        culture_war_companies : List[str]
            List of ticker symbols
        start_date : str
            Start date for filing search
        end_date : str
            End date for filing search
        save_csv : bool
            Whether to save to CSV
            
        Returns:
        --------
        pd.DataFrame : Complete Form 4 transaction dataset
        """
        all_transactions = []
        
        for ticker in culture_war_companies:
            print(f"\nProcessing {ticker}...")
            
            # Get CIK
            cik = self.get_company_cik(ticker)
            if not cik:
                continue
            
            # Download filing list
            filings = self.download_form4_filings(ticker, cik, start_date, end_date)
            
            # Parse each filing
            for filing in filings:
                transactions = self.parse_form4_xml(filing['filing_url'])
                
                # Add filing metadata to each transaction
                for trans in transactions:
                    trans['ticker'] = ticker
                    trans['cik'] = cik
                    trans['filing_date'] = filing['filing_date']
                    trans['accession_number'] = filing['accession_number']
                    trans['filing_url'] = filing['filing_url']
                    
                all_transactions.extend(transactions)
                
                # Rate limiting
                time.sleep(0.15)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_transactions)
        
        if len(df) > 0:
            # Clean and format
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['filing_date'] = pd.to_datetime(df['filing_date'])
            df['transaction_value'] = df['shares'] * df['price_per_share']
            
            # Sort by transaction date
            df = df.sort_values('transaction_date')
            
            # Save to CSV
            if save_csv:
                output_file = os.path.join(
                    self.output_dir,
                    f'form4_transactions_{start_date}_to_{end_date}.csv'
                )
                df.to_csv(output_file, index=False)
                print(f"\n✓ Saved {len(df)} transactions to {output_file}")
        
        return df


def load_culture_war_companies(culture_war_data: pd.DataFrame) -> List[str]:
    """
    Extract unique company tickers from culture war dataset.
    
    Parameters:
    -----------
    culture_war_data : pd.DataFrame
        Your culture war events dataset
        
    Returns:
    --------
    List[str] : Unique ticker symbols
    """
    # Check for various possible column names (case-insensitive)
    possible_ticker_cols = ['Ticker', 'ticker', 'TICKER', 'Symbol', 'symbol']
    possible_company_cols = ['Company', 'company', 'COMPANY', 'company_name']
    
    # Find ticker column
    ticker_col = None
    for col in possible_ticker_cols:
        if col in culture_war_data.columns:
            ticker_col = col
            break
    
    if ticker_col:
        tickers = culture_war_data[ticker_col].unique().tolist()
        # Filter out NaN, None, 'Private', and other non-ticker values
        tickers = [
            t for t in tickers 
            if pd.notna(t) and 
            str(t).strip() not in ['', 'Private', 'N/A', 'NA', 'None']
        ]
        return tickers
    
    # Try company column as fallback
    for col in possible_company_cols:
        if col in culture_war_data.columns:
            companies = culture_war_data[col].unique().tolist()
            print(f"Warning: Found company names but not tickers. You'll need to map company names to tickers.")
            return [c for c in companies if pd.notna(c)]
    
    # If nothing found, show available columns
    print(f"Available columns: {culture_war_data.columns.tolist()}")
    raise ValueError("Cannot find ticker or company column in culture war data")

#--- DATA DICTIONARY --------
def load_data():
    """
    Master data loading function for all research datasets.
    Returns dictionary with standardized keys for consistent access across modules.
    """
    
    # Import/load all datasets with proper file paths
    culture_war_data = import_culture_war_data('/Users/ashleyroseboro/Signalsandsystems/Culture_War_Companies_160_fullmeta.csv')
    stock_data = load_stock_data()
    vix_df = load_vix_data()
    ff_factors = load_fama_french_factors()
    form4_data = load_form4_data()  # Added Form 4 insider trading data
    
    # Build comprehensive data dictionary
    full_dictionary = {
        "culturewardata": culture_war_data,
        "stockdata": stock_data,
        "vixdata": vix_df,
        "ff_factors": ff_factors,
        "form4data": form4_data,  # Added to dictionary
    }
    
    return full_dictionary


def load_vix_data(file_path='vix_data_2000_2025.csv'):
    """
    Load VIX data from CSV file.
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(SCRIPT_DIR, file_path)
    
    if os.path.exists(file_path):
        vix_df = pd.read_csv(file_path)
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        return vix_df
    else:
        print(f"VIX data file not found at {file_path}")
        return None


def load_stock_data(tickers=None, start_date='2000-01-01', end_date='2025-12-31'):
    """
    Load stock data for given tickers. If no tickers provided, loads from culture war data.
    """
    if tickers is None:
        # Load tickers from culture war data
        culture_war_data = import_culture_war_data('/Users/ashleyroseboro/Signalsandsystems/Culture_War_Companies_160_fullmeta.csv')
        tickers = culture_war_data['Ticker'].unique().tolist()
    
    stock_data = get_stock_data(tickers, start_date=start_date, end_date=end_date)
    return stock_data


def load_fama_french_factors(
    start_date='2000-01-01',
    end_date=None,
    frequency='daily',
    cache_path='./data/fama_french'
):
    """
    Load Fama-French factor data with caching.
    """
    import pandas as pd
    import pandas_datareader as pdr
    from datetime import datetime
    import os
    
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Check for cached data
    cache_file = os.path.join(cache_path, f'ff_factors_{frequency}_{start_date}_{end_date}.pkl')
    
    if os.path.exists(cache_file):
        print(f"Loading cached Fama-French data from {cache_file}")
        return pd.read_pickle(cache_file)
    
    # Download fresh data
    print("Downloading Fama-French factor data...")
    
    freq_map = {
        'daily': 'F-F_Research_Data_Factors_daily',
        'monthly': 'F-F_Research_Data_Factors'
    }
    
    try:
        # Download FF3
        ff3 = pdr.DataReader(
            freq_map[frequency],
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        
        # Download FF5
        ff5_name = 'F-F_Research_Data_5_Factors_2x3_daily' if frequency == 'daily' else 'F-F_Research_Data_5_Factors_2x3'
        ff5 = pdr.DataReader(
            ff5_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        
        # Download Momentum
        mom_name = 'F-F_Momentum_Factor_daily' if frequency == 'daily' else 'F-F_Momentum_Factor'
        mom = pdr.DataReader(
            mom_name,
            'famafrench',
            start=start_date,
            end=end_date
        )[0]
        
        # Combine into single DataFrame
        ff_data = {
            'FF3': ff3,
            'FF5': ff5,
            'Momentum': mom
        }
        
        # Cache the data
        os.makedirs(cache_path, exist_ok=True)
        pd.to_pickle(ff_data, cache_file)
        print(f"Cached data to {cache_file}")
        
        return ff_data
        
    except Exception as e:
        print(f"Error loading Fama-French data: {e}")
        return None


def load_form4_data(
    cache_file='./sec_form4_data/form4_transactions_2000-01-01_to_2025-12-31.csv',
    refresh=False
):
    """
    Load Form 4 insider trading data. Downloads if not cached or refresh=True.
    
    Parameters:
    -----------
    cache_file : str
        Path to cached Form 4 CSV file
    refresh : bool
        If True, re-download data even if cache exists
        
    Returns:
    --------
    pd.DataFrame : Form 4 transaction data
    """
    import os
    
    # Check if cached file exists
    if os.path.exists(cache_file) and not refresh:
        print(f"Loading cached Form 4 data from {cache_file}")
        form4_df = pd.read_csv(cache_file)
        form4_df['transaction_date'] = pd.to_datetime(form4_df['transaction_date'])
        form4_df['filing_date'] = pd.to_datetime(form4_df['filing_date'])
        return form4_df
    
    # Download fresh data
    print("Downloading Form 4 data from SEC EDGAR...")
    
    # Initialize downloader
    downloader = Form4Downloader(output_dir='./sec_form4_data')
    
    # Load culture war companies
    culture_war_data = import_culture_war_data('/Users/ashleyroseboro/Signalsandsystems/Culture_War_Companies_160_fullmeta.csv')
    companies = culture_war_data['Ticker'].unique().tolist()
    
    # Download and parse all Form 4 filings
    form4_df = downloader.build_form4_dataset(
        culture_war_companies=companies,
        start_date='2000-01-01',
        end_date='2025-12-31',
        save_csv=True
    )
    
    return form4_df


# Example usage in your analysis pipeline
if __name__ == "__main__":
    # Load all data
    data_dict = load_data()
    
    # Access specific datasets
    culture_wars = data_dict["culturewardata"]
    stocks = data_dict["stockdata"]
    vix = data_dict["vixdata"]
    ff_factors = data_dict["ff_factors"]
    form4 = data_dict["form4data"]  # New: Insider trading data
    
    # Access specific factor models
    ff3_factors = ff_factors['FF3']
    ff5_factors = ff_factors['FF5']
    momentum = ff_factors['Momentum']
    
    # Print summary
    print("\n=== Data Dictionary Summary ===")
    for key, value in data_dict.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for subkey, df in value.items():
                print(f"  {subkey}: {df.shape if hasattr(df, 'shape') else 'N/A'}")
        else:
            print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    
        # Essay 3 specific analysis example
        if form4 is not None and len(form4) > 0:
            print("\n=== Form 4 Insider Trading Summary ===")
            print(f"Total transactions: {len(form4)}")
            print(f"Date range: {form4['transaction_date'].min()} to {form4['transaction_date'].max()}")
            print(f"Unique companies: {form4['ticker'].nunique()}")
            print(f"Unique insiders: {form4['owner_name'].nunique()}")
            print("\nTransaction types:")
            print(form4['transaction_code'].value_counts())
            print("\nTop 10 most active insiders:")
            print(form4['owner_name'].value_counts().head(10))
    
    
    # ===== FORM 4 USAGE EXAMPLE =====
    if __name__ == "__main__":
        # Initialize downloader
        downloader = Form4Downloader(output_dir='./sec_form4_data')
        
        # Load your culture war data
        data_dict = load_data()  # Your existing function
        culture_war_data = data_dict['culturewardata']
        
        # Get unique companies
        companies = load_culture_war_companies(culture_war_data)
        print(f"Found {len(companies)} unique companies in culture war dataset")
        
        # Download Form 4 data for all companies
        form4_df = downloader.build_form4_dataset(
            culture_war_companies=companies,
            start_date='2000-01-01',
            end_date='2025-12-31',
            save_csv=True
        )
        
        # Print summary
print("\n=== Data Dictionary Summary ===")
for key, value in data_dict.items():
    print(f"\n{key}:")
    if isinstance(value, dict):
        for subkey, df in value.items():
            print(f"  {subkey}: {df.shape if hasattr(df, 'shape') else 'N/A'}")
    else:
        print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")

# Essay 3 specific analysis example
print("\n=== Form 4 Insider Trading Summary ===")
if form4 is not None and len(form4) > 0:
    print(f"Total transactions: {len(form4)}")
    print(f"Date range: {form4['transaction_date'].min()} to {form4['transaction_date'].max()}")
    print(f"Unique companies: {form4['ticker'].nunique()}")
    print(f"Unique insiders: {form4['owner_name'].nunique()}")
    print("\nTransaction types:")
    print(form4['transaction_code'].value_counts())
    print("\nTop 10 most active insiders:")
    print(form4['owner_name'].value_counts().head(10))
else:
    print("No Form 4 transactions found. Possible issues:")
    print("  1. Companies may not have filed Form 4s in the date range")
    print("  2. CIK lookup may have failed for most companies")
    print("  3. SEC API may be rate-limiting requests")
    print("\nTry:")
    print("  - Check if ./sec_form4_data/ directory has any files")
    print("  - Verify a few tickers manually at https://www.sec.gov/edgar/searchedgar/companysearch")
    print("  - Run with a smaller subset of companies first")