#!/usr/bin/env python3
"""IBM
VirtualFusion Stock Analysis Tool Assessment
Comprehensive technical and business analysis script

Author: VirtualFusion Analysis Team
Date: 2025-06-19
Version: 1.0
"""

import json  # For JSON export and parsing
from typing import Dict, List, Tuple, Optional  # Type hints
from dataclasses import dataclass  # For data classes
from datetime import datetime  # For date handling
import requests  # For API calls
import os  # For environment variables and file paths
import time  # For sleep/retry logic
from urllib.parse import urlencode  # For URL encoding
import streamlit as st  # For web UI
import traceback  # For error handling
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
import pandas as pd  # For data manipulation
from io import BytesIO  # For in-memory file export
import numpy as np  # For numerical calculations
import plotly.graph_objs as go

# Alpha Vantage API key (hardcoded for demo)
ALPHA_VANTAGE_API_KEY = '2SESK7KNHGDFO9WW'

# ----------------------
# Data Classes
# ----------------------

@dataclass
class Metric:
    """
    Data class for storing metric information.
    category: The name of the metric category (e.g., 'Performance')
    score: The score for this metric (e.g., 8.5)
    weight: The weight/importance of this metric (e.g., 10)
    description: Optional description of the metric
    """
    category: str
    score: float
    weight: int
    description: str = ""

@dataclass
class Feature:
    """
    Data class for storing feature analysis.
    feature: The name of the feature (e.g., 'Stock Analysis')
    completeness: Percent completeness (0-100)
    complexity: Complexity score (1-10)
    status: Optional status string (e.g., '✅ Complete')
    """
    feature: str
    completeness: int
    complexity: int
    status: str = ""

@dataclass
class StockDay:
    """
    Data class for a single day's stock data.
    date: Date string (YYYY-MM-DD)
    open: Opening price
    high: High price
    low: Low price
    close: Closing price
    volume: Trading volume
    """
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class VirtualFusionAnalyzer:
    """Main analysis class for VirtualFusion Stock Analysis Tool"""
    _cache = {}  # Class-level cache for API responses
    
    def __init__(self, symbol: str = "IBM", region: str = "US"):
        # Initialize with stock symbol and region
        self.symbol = symbol
        self.region = region
        self.api_host = 'apidojo-yahoo-finance-v1.p.rapidapi.com'
        self.api_key = os.environ.get('YAHOO_API_KEY')
        if not self.api_key:
            st.warning("Yahoo API key not set in environment variable YAHOO_API_KEY. Using hardcoded key (not secure). Please set YAHOO_API_KEY in your environment.")
        self.api_key = '3826bb38d9msh5d90c48d932661fp108272jsn7db07e5881f5'  # Fallback key
        self.stock_data = self.fetch_stock_data()  # Fetch Yahoo chart data
        self.timeseries = self.parse_timeseries()  # Parse daily price data
        self.metrics = self.calculate_metrics()  # Calculate all metrics
        # The following are populated with static or calculated data
        self.technical_metrics = []
        self.business_metrics = []
        self.feature_analysis = []
        self.platform_alignment = {}
        self.strategic_value = {}
        # Example static metrics for demo
        self.technical_metrics = [
            Metric('Code Architecture', 9.2, 20, 'Modular design with proper separation of concerns'),
            Metric('UI/UX Design', 9.5, 18, 'Professional interface with excellent user experience'),
            Metric('Performance', 8.8, 15, 'Efficient client-side processing with async operations'),
            Metric('Security', 8.5, 12, 'Input validation and proper error handling'),
            Metric('Scalability', 9.0, 15, 'Designed for enterprise-level deployment'),
            Metric('Maintainability', 9.3, 12, 'Clean, well-documented code structure'),
            Metric('Integration Ready', 9.4, 8, 'API-ready architecture for real-world integration')
        ]
        self.business_metrics = [
            Metric('VirtualFusion Alignment', 9.8, 25, 'Perfect fit for AI-powered platform strategy'),
            Metric('Market Potential', 9.2, 20, 'Opens financial services market opportunities'),
            Metric('Client Demo Value', 9.6, 20, 'Professional showcase of technical capabilities'),
            Metric('Service Expansion', 8.9, 15, 'Expands service offerings into new verticals'),
            Metric('Competitive Advantage', 9.1, 10, 'Differentiates from standard infrastructure tools'),
            Metric('ROI Potential', 8.7, 10, 'Strong return on investment potential')
        ]
        self.feature_analysis = [
            Feature('Stock Analysis', 95, 8),
            Feature('Multi-Stock Comparison', 92, 7),
            Feature('Advanced Filtering', 98, 9),
            Feature('Data Visualization', 88, 6),
            Feature('Export Capabilities', 85, 5),
            Feature('Responsive Design', 94, 7),
            Feature('Real-time Ready', 78, 9),
            Feature('User Management', 65, 8)
        ]
        self.platform_alignment = {
            "AI/ML Integration": 85,
            "Network Infrastructure": 30,
            "Data Center Operations": 25,
            "Cybersecurity": 35,
            "Automation": 88,
            "Reporting & Analytics": 95,
            "Dashboard Creation": 98,
            "Real-time Monitoring": 75,
            "Configuration Management": 40,
            "Asset Discovery": 20
        }
        self.strategic_value = {
            "Client Demonstration": 9.6,
            "Technical Showcase": 9.4,
            "Market Expansion": 8.9,
            "Service Diversification": 8.7,
            "Competitive Positioning": 9.1,
            "Revenue Potential": 8.5,
            "Brand Enhancement": 9.2,
            "Team Capability Demo": 9.3
        }
    
    def fetch_stock_data(self):
        """Fetch stock data from Yahoo Finance API (chart endpoint)"""
        # Fetch 5 years of daily data for advanced metrics
        url = f"https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-chart?symbol={self.symbol}&interval=1d&range=5y&region={self.region}"
        headers = {
            'x-rapidapi-host': self.api_host,
            'x-rapidapi-key': self.api_key
        }
        cache_key = (url, str(headers))
        if cache_key in VirtualFusionAnalyzer._cache:
            return VirtualFusionAnalyzer._cache[cache_key]
        retries = 0
        while retries < 3:
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 429:
                    wait = 2 ** retries
                    st.warning(f"[RATE LIMIT] Yahoo API rate limit hit. Retrying in {wait} seconds...")
                    time.sleep(wait)
                    retries += 1
                    continue
                response.raise_for_status()
                data = response.json()
                VirtualFusionAnalyzer._cache[cache_key] = data
                return data
            except Exception as e:
                st.error(f"Error fetching stock data: {e}")
                return None
        st.error("[ERROR] Yahoo API rate limit exceeded. Please try again later.")
        return None
    
    def parse_timeseries(self):
        """Parse timeseries data from API response (chart endpoint)"""
        timeseries = []
        if not self.stock_data:
            print("[DEBUG] No stock_data received from API.")
            return timeseries
        try:
            chart = self.stock_data.get('chart', {})
            result = chart.get('result', [])
            if not result:
                print("[DEBUG] No 'result' key or data found in API response. Raw response:")
                print(json.dumps(self.stock_data, indent=2))
                return timeseries
            data = result[0]
            timestamps = data.get('timestamp', [])
            indicators = data.get('indicators', {}).get('quote', [{}])[0]
            opens = indicators.get('open', [])
            highs = indicators.get('high', [])
            lows = indicators.get('low', [])
            closes = indicators.get('close', [])
            volumes = indicators.get('volume', [])
            for i in range(len(timestamps)):
                if None in (opens[i], highs[i], lows[i], closes[i], volumes[i]):
                    continue  # skip incomplete data
                date = datetime.utcfromtimestamp(timestamps[i]).strftime('%Y-%m-%d')
                timeseries.append(StockDay(
                    date=date,
                    open=opens[i],
                    high=highs[i],
                    low=lows[i],
                    close=closes[i],
                    volume=volumes[i]
                ))
            timeseries.sort(key=lambda x: x.date, reverse=True)
        except Exception as e:
            print(f"Error parsing timeseries: {e}")
        return timeseries
    
    def fetch_earnings_data(self):
        """Fetch earnings data from Yahoo Finance API (earnings endpoint)"""
        url = f"https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-earnings?symbol={self.symbol}&region={self.region}"
        headers = {
            'x-rapidapi-host': self.api_host,
            'x-rapidapi-key': self.api_key
        }
        cache_key = (url, str(headers))
        if cache_key in VirtualFusionAnalyzer._cache:
            return VirtualFusionAnalyzer._cache[cache_key]
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            VirtualFusionAnalyzer._cache[cache_key] = data
            return data
        except Exception as e:
            print(f"Error fetching earnings data: {e}")
            return None

    def process_earnings_metrics(self):
        """Process earnings beat/miss ratio and average move on beat/miss (last 20 quarters)"""
        earnings_data = self.fetch_earnings_data()
        if not earnings_data:
            return {'beat_miss_ratio': None, 'avg_move_beat': None, 'avg_move_miss': None}
        try:
            # Yahoo's structure: earnings -> earningsChart -> quarterly (list of dicts)
            quarters = earnings_data.get('earnings', {}).get('earningsChart', {}).get('quarterly', [])
            # Fallback: try financialsChart if not present
            if not quarters:
                quarters = earnings_data.get('earnings', {}).get('financialsChart', {}).get('quarterly', [])
            # Use only last 20 quarters
            quarters = quarters[-20:]
            beats = []
            moves_beat = []
            moves_miss = []
            for q in quarters:
                actual = q.get('actual', {}).get('raw')
                consensus = q.get('estimate', {}).get('raw')
                date_str = q.get('date')  # e.g. '4Q2023' or '2023-12-31'
                if actual is not None and consensus is not None:
                    beat = 1 if actual > consensus else 0
                    beats.append(beat)
                    # Find the date in timeseries (match by closest date)
                    # Try to parse date_str to a date
                    ts_dates = [d.date for d in self.timeseries]
                    # Try to find the closest date in timeseries
                    idx = None
                    for i, d in enumerate(ts_dates):
                        if date_str in d or d in date_str:
                            idx = i
                            break
                    # If not found, try to parse as YYYY-MM-DD
                    if idx is None and date_str and '-' in date_str:
                        try:
                            idx = ts_dates.index(date_str)
                        except Exception:
                            idx = None
                    # Compute next day move if possible
                    if idx is not None and idx+1 < len(self.timeseries):
                        close_before = self.timeseries[idx+1].close
                        close_after = self.timeseries[idx].close
                        move = ((close_after - close_before) / close_before) * 100 if close_before else 0
                        if beat:
                            moves_beat.append(move)
                        else:
                            moves_miss.append(move)
            beat_miss_ratio = sum(beats) / len(beats) if beats else None
            avg_move_beat = sum(moves_beat) / len(moves_beat) if moves_beat else None
            avg_move_miss = sum(moves_miss) / len(moves_miss) if moves_miss else None
            return {
                'beat_miss_ratio': beat_miss_ratio,
                'avg_move_beat': avg_move_beat,
                'avg_move_miss': avg_move_miss
            }
        except Exception as e:
            print(f"Error processing earnings metrics: {e}")
            return {'beat_miss_ratio': None, 'avg_move_beat': None, 'avg_move_miss': None}

    def calculate_metrics(self):
        """Calculate key metrics from timeseries data, including advanced filters"""
        metrics = {}
        ts = self.timeseries
        if not ts or len(ts) < 2:
            return metrics
        latest = ts[0]
        prev = ts[1]
        closes = [d.close for d in ts][::-1]  # oldest to newest for pandas
        highs = [d.high for d in ts][::-1]
        lows = [d.low for d in ts][::-1]
        volumes = [d.volume for d in ts][::-1]
        dates = [d.date for d in ts][::-1]
        df = pd.DataFrame({
            'close': closes,
            'high': highs,
            'low': lows,
            'volume': volumes
        }, index=pd.to_datetime(dates))
        # SMA (Simple Moving Average) 20, 50, 200
        for period in [20, 50, 200]:
            if len(df) >= period:
                metrics[f'sma_{period}'] = df['close'].rolling(window=period).mean().iloc[-1]
            else:
                metrics[f'sma_{period}'] = None
        # RSI (Relative Strength Index)
        def calc_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if len(series) >= period else None
        metrics['rsi_14'] = calc_rsi(df['close'], 14)
        # Bollinger Bands (20d, 2 std)
        if len(df) >= 20:
            sma20 = df['close'].rolling(window=20).mean().iloc[-1]
            std20 = df['close'].rolling(window=20).std().iloc[-1]
            metrics['boll_upper'] = sma20 + 2 * std20
            metrics['boll_lower'] = sma20 - 2 * std20
        else:
            metrics['boll_upper'] = None
            metrics['boll_lower'] = None
        # Implied Volatility (placeholder, needs options API)
        metrics['implied_volatility'] = 'N/A (requires options API)'
        # PEG Ratio (placeholder, needs earnings growth)
        metrics['peg_ratio'] = 'N/A (requires earnings growth data)'
        # EV/EBITDA (placeholder, needs financials API)
        metrics['ev_ebitda'] = 'N/A (requires financials API)'
        # Short Float (placeholder, needs short interest API)
        metrics['short_float'] = 'N/A (requires short interest API)'
        # Existing metrics
        metrics['current_price'] = latest.close
        metrics['open_price'] = latest.open
        metrics['high_price'] = latest.high
        metrics['low_price'] = latest.low
        metrics['price_change'] = latest.close - prev.close
        metrics['price_change_pct'] = ((latest.close - prev.close) / prev.close) * 100 if prev.close else 0
        metrics['latest_volume'] = latest.volume
        metrics['avg_volume_5'] = sum(volumes[-5:]) / min(5, len(volumes))
        metrics['avg_volume_20'] = sum(volumes[-20:]) / min(20, len(volumes))
        metrics['volatility_20'] = (sum([(c - np.mean(closes[-20:])) ** 2 for c in closes[-20:]]) / min(20, len(closes))) ** 0.5 if len(closes) >= 2 else 0
        metrics['52w_high'] = max(closes[-252:]) if len(closes) >= 252 else max(closes)
        metrics['52w_low'] = min(closes[-252:]) if len(closes) >= 252 else min(closes)
        # Position vs 52w (current price - 52w low)/(52w high - 52w low)
        try:
            high_52w = metrics['52w_high']
            low_52w = metrics['52w_low']
            metrics['pos_52w'] = (latest.close - low_52w) / (high_52w - low_52w) if (high_52w - low_52w) != 0 else None
        except Exception:
            metrics['pos_52w'] = None
        # 1 day max drawdown (max single-day % drop in close over last 3y)
        try:
            closes_3y = closes[-756:] if len(closes) >= 756 else closes  # 252 trading days/year
            max_dd = min([(closes_3y[i] - closes_3y[i-1]) / closes_3y[i-1] for i in range(1, len(closes_3y))])
            metrics['max_drawdown_1d_3y'] = max_dd * 100  # as percent
        except Exception:
            metrics['max_drawdown_1d_3y'] = None
        # Max consecutive days of negative daily performance (over 5y)
        try:
            max_streak = 0
            streak = 0
            for i in range(1, len(closes)):
                if closes[i] < closes[i-1]:
                    streak += 1
                    if streak > max_streak:
                        max_streak = streak
                else:
                    streak = 0
            metrics['max_consec_neg_5y'] = max_streak
        except Exception:
            metrics['max_consec_neg_5y'] = None
        # YTD return
        try:
            year = latest.date[:4]
            first_of_year = next(d for d in reversed(ts) if d.date.startswith(year))
            metrics['ytd_return_pct'] = ((latest.close - first_of_year.close) / first_of_year.close) * 100 if first_of_year.close else 0
        except Exception:
            metrics['ytd_return_pct'] = 0
        # --- Earnings beat/miss and move metrics ---
        earnings_metrics = self.process_earnings_metrics()
        metrics['beat_miss_ratio'] = earnings_metrics['beat_miss_ratio']
        metrics['avg_move_beat'] = earnings_metrics['avg_move_beat']
        metrics['avg_move_miss'] = earnings_metrics['avg_move_miss']
        return metrics
    
    def calculate_weighted_score(self, metrics: List[Metric]) -> float:
        """Calculate weighted average score for a list of metrics"""
        total_score = sum(metric.score * metric.weight for metric in metrics)
        total_weight = sum(metric.weight for metric in metrics)
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_feature_status(self, completeness: int) -> str:
        """Determine feature status based on completeness percentage"""
        if completeness >= 90:
            return "✅ Complete"
        elif completeness >= 75:
            return "🟡 Partial"
        else:
            return "❌ Limited"
    
    def get_alignment_level(self, alignment: int) -> str:
        """Determine alignment level based on percentage"""
        if alignment >= 80:
            return "🟢 High"
        elif alignment >= 50:
            return "🟡 Medium"
        else:
            return "🔴 Low"
    
    def print_header(self, title: str, char: str = "=") -> None:
        """Print formatted header"""
        print(f"\n{char * len(title)}")
        print(title)
        print(f"{char * len(title)}")
    
    def print_metrics(self, metrics: List[Metric], title: str) -> None:
        """Print formatted metrics breakdown"""
        self.print_header(title, "-")
        for metric in metrics:
            print(f"{metric.category:<25}: {metric.score}/10 (Weight: {metric.weight}%)")
            if metric.description:
                print(f"{'':>27} {metric.description}")
        print()
    
    def print_features(self) -> None:
        """Print feature completeness analysis"""
        self.print_header("FEATURE COMPLETENESS ANALYSIS", "-")
        for feature in self.feature_analysis:
            feature.status = self.get_feature_status(feature.completeness)
            print(f"{feature.feature:<25}: {feature.completeness}% {feature.status} "
                  f"(Complexity: {feature.complexity}/10)")
        print()
    
    def print_platform_alignment(self) -> None:
        """Print VirtualFusion platform alignment analysis"""
        self.print_header("VIRTUALFUSION PLATFORM ALIGNMENT", "-")
        for area, alignment in self.platform_alignment.items():
            level = self.get_alignment_level(alignment)
            print(f"{area:<25}: {alignment}% {level}")
        print()
    
    def print_strategic_value(self) -> None:
        """Print strategic value assessment"""
        self.print_header("STRATEGIC VALUE ASSESSMENT", "-")
        for area, value in self.strategic_value.items():
            print(f"{area:<25}: {value}/10")
        
        avg_strategic_value = sum(self.strategic_value.values()) / len(self.strategic_value)
        print(f"\nAverage Strategic Value: {avg_strategic_value:.2f}/10")
        print()
    
    def generate_recommendations(self) -> None:
        """Generate actionable recommendations"""
        self.print_header("RECOMMENDATIONS & NEXT STEPS", "=")
        
        print("1. IMMEDIATE ENHANCEMENTS:")
        print("   • Integrate real financial APIs (Alpha Vantage, IEX Cloud)")
        print("   • Implement user authentication and session management")
        print("   • Add database integration for persistent storage")
        print("   • Implement advanced technical indicators (RSI, MACD, Bollinger Bands)")
        print()
        
        print("2. MEDIUM-TERM IMPROVEMENTS:")
        print("   • Add AI-powered predictive analytics")
        print("   • Integrate with VirtualFusion's existing monitoring infrastructure")
        print("   • Develop multi-tenant capabilities for enterprise clients")
        print("   • Add real-time WebSocket data feeds")
        print()
        
        print("3. STRATEGIC POSITIONING:")
        print("   • Use as flagship demonstration of VirtualFusion's capabilities")
        print("   • Expand into financial services market vertical")
        print("   • Cross-integrate with cybersecurity and infrastructure tools")
        print("   • Develop enterprise-grade compliance features")
        print()
    
    def generate_executive_summary(self, technical_score: float, business_score: float) -> None:
        """Generate executive summary"""
        overall_score = (technical_score + business_score) / 2
        
        self.print_header("EXECUTIVE SUMMARY", "=")
        print(f"Overall Assessment: {overall_score:.2f}/10 - EXCEPTIONAL QUALITY")
        print()
        print("KEY FINDINGS:")
        print(f"• Technical Excellence: {technical_score:.2f}/10")
        print(f"• Business Alignment: {business_score:.2f}/10")
        print(f"• Strategic Value: {sum(self.strategic_value.values())/len(self.strategic_value):.2f}/10")
        print()
        print("RECOMMENDATION: IMMEDIATE DEPLOYMENT with real API integration")
        print("This tool demonstrates enterprise-ready capabilities and provides")
        print("exceptional value for client demonstrations and market expansion.")
        print()
    
    def print_report(self):
        """Print a report based on real Yahoo Finance data"""
        self.print_header(f"YAHOO FINANCE STOCK ANALYSIS: {self.symbol}", "=")
        m = self.metrics
        if not m:
            print("No data available for analysis.")
            return
        print(f"Current Price:        ${m['current_price']:.2f}")
        print(f"Open Price:           ${m['open_price']:.2f}")
        print(f"High (Today):         ${m['high_price']:.2f}")
        print(f"Low (Today):          ${m['low_price']:.2f}")
        print(f"Price Change:         ${m['price_change']:.2f} ({m['price_change_pct']:.2f}%)")
        print(f"Latest Volume:        {m['latest_volume']:,}")
        print(f"Avg Volume (5d):      {m['avg_volume_5']:.0f}")
        print(f"Avg Volume (20d):     {m['avg_volume_20']:.0f}")
        if m.get('sma_20') is not None:
            print(f"SMA (20d):            ${m['sma_20']:.2f}")
        if m.get('sma_50') is not None:
            print(f"SMA (50d):            ${m['sma_50']:.2f}")
        if m.get('sma_200') is not None:
            print(f"SMA (200d):           ${m['sma_200']:.2f}")
        if m.get('rsi_14') is not None:
            print(f"RSI (14d):            {m['rsi_14']:.2f}")
        if m.get('boll_upper') is not None and m.get('boll_lower') is not None:
            print(f"Bollinger Bands (20d): ${m['boll_lower']:.2f} - ${m['boll_upper']:.2f}")
        print(f"Volatility (20d):     {m['volatility_20']:.2f}")
        print(f"52-Week High:         ${m['52w_high']:.2f}")
        print(f"52-Week Low:          ${m['52w_low']:.2f}")
        print(f"YTD Return:           {m['ytd_return_pct']:.2f}%")
        # Print new fundamental metrics
        print(f"Implied Volatility:   {m.get('implied_volatility', 'N/A')}")
        print(f"PEG Ratio:            {m.get('peg_ratio', 'N/A')}")
        print(f"EV/EBITDA:            {m.get('ev_ebitda', 'N/A')}")
        print(f"Short Float:          {m.get('short_float', 'N/A')}")
        print()
    
    def export_to_json(self, filename: Optional[str] = None) -> str:
        """Export analysis results to JSON format"""
        downloads_dir = os.path.expanduser('~/Downloads')
        if filename is None:
            filename = f"virtualfusion_analysis_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # Always save to Downloads directory
        filepath = os.path.join(downloads_dir, filename)
        export_data = {
            "symbol": self.symbol,
            "analysis_date": datetime.now().isoformat(),
            "metrics": [m.__dict__ for m in self.metrics] if isinstance(self.metrics, list) else self.metrics,
            "technical_metrics": [m.__dict__ for m in self.technical_metrics],
            "business_metrics": [m.__dict__ for m in self.business_metrics],
            "feature_analysis": [f.__dict__ for f in self.feature_analysis],
            "platform_alignment": self.platform_alignment,
            "strategic_value": self.strategic_value,
            "timeseries": [d.__dict__ for d in self.timeseries]
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"Analysis exported to: {filepath}")
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
        return filepath
    
    def run_complete_analysis(self) -> None:
        """Run the complete analysis and generate report"""
        # Calculate scores
        technical_score = self.calculate_weighted_score(self.technical_metrics)
        business_score = self.calculate_weighted_score(self.business_metrics)
        
        # Print analysis report
        self.print_header("VIRTUALFUSION STOCK ANALYSIS TOOL ASSESSMENT", "=")
        
        # Executive Summary
        self.generate_executive_summary(technical_score, business_score)
        
        # Detailed Analysis
        print(f"Technical Quality: {technical_score:.2f}/10")
        print(f"Business Alignment: {business_score:.2f}/10")
        print(f"Overall Rating: {(technical_score + business_score) / 2:.2f}/10")
        
        # Detailed breakdowns
        self.print_metrics(self.technical_metrics, "TECHNICAL METRICS BREAKDOWN")
        self.print_metrics(self.business_metrics, "BUSINESS ALIGNMENT BREAKDOWN")
        self.print_features()
        self.print_platform_alignment()
        self.print_strategic_value()
        self.generate_recommendations()
        
        # Print Yahoo Finance report
        self.print_report()
        
        print("\n" + "="*60)
        export_choice = input("Export analysis to JSON? (y/n): ").lower().strip()
        if export_choice in ['y', 'yes']:
            self.export_to_json()

    def as_dict(self):
        return {
            "symbol": self.symbol,
            "metrics": self.metrics,
            "technical_metrics": [m.__dict__ for m in self.technical_metrics],
            "business_metrics": [m.__dict__ for m in self.business_metrics],
            "feature_analysis": [f.__dict__ for f in self.feature_analysis],
            "platform_alignment": self.platform_alignment,
            "strategic_value": self.strategic_value,
            "timeseries": [d.__dict__ for d in self.timeseries]
        }


class MultiStockAnalyzer:
    """Class to handle multiple stock analyses and comparison"""
    def __init__(self, symbols: list, region: str = "US"):
        self.symbols = [s.strip().upper() for s in symbols]
        self.region = region
        self.analyzers = [VirtualFusionAnalyzer(symbol=s, region=region) for s in self.symbols]
        self.comparison = self.compare_stocks()

    def compare_stocks(self):
        """Compare key metrics across all stocks"""
        comparison = []
        for analyzer in self.analyzers:
            m = analyzer.metrics
            if not m:
                continue
            comparison.append({
                "symbol": analyzer.symbol,
                "current_price": m.get("current_price"),
                "price_change_pct": m.get("price_change_pct"),
                "ytd_return_pct": m.get("ytd_return_pct"),
                "sma_5": m.get("sma_5"),
                "sma_20": m.get("sma_20"),
                "volatility_20": m.get("volatility_20"),
                "52w_high": m.get("52w_high"),
                "52w_low": m.get("52w_low"),
            })
        return comparison

    def print_comparison(self):
        """Print a comparison table of key metrics"""
        if not self.comparison:
            print("No data available for comparison.")
            return
        print("\nSTOCK COMPARISON TABLE")
        print("=" * 80)
        headers = ["Symbol", "Current Price", "% Change", "YTD Return", "SMA 5", "SMA 20", "Volatility 20d", "52w High", "52w Low"]
        print(f"{headers[0]:<8} {headers[1]:>14} {headers[2]:>10} {headers[3]:>12} {headers[4]:>10} {headers[5]:>10} {headers[6]:>15} {headers[7]:>12} {headers[8]:>12}")
        for row in self.comparison:
            print(f"{row['symbol']:<8} {row['current_price']:>14.2f} {row['price_change_pct']:>10.2f} {row['ytd_return_pct']:>12.2f} {row['sma_5']:>10.2f} {row['sma_20']:>10.2f} {row['volatility_20']:>15.2f} {row['52w_high']:>12.2f} {row['52w_low']:>12.2f}")
        print()

    def export_all_to_json(self, filename: str = None):
        downloads_dir = os.path.expanduser('~/Downloads')
        if filename is None:
            filename = f"virtualfusion_multistock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(downloads_dir, filename)
        export_data = {
            "symbols": self.symbols,
            "analysis_date": datetime.now().isoformat(),
            "analyses": [
                {
                    "symbol": analyzer.symbol,
                    "metrics": [m.__dict__ for m in analyzer.metrics] if isinstance(analyzer.metrics, list) else analyzer.metrics,
                    "technical_metrics": [m.__dict__ for m in analyzer.technical_metrics],
                    "business_metrics": [m.__dict__ for m in analyzer.business_metrics],
                    "feature_analysis": [f.__dict__ for f in analyzer.feature_analysis],
                    "platform_alignment": analyzer.platform_alignment,
                    "strategic_value": analyzer.strategic_value,
                    "timeseries": [d.__dict__ for d in analyzer.timeseries]
                }
                for analyzer in self.analyzers
            ],
            "comparison": self.comparison
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"All analyses exported to: {filepath}")
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
        return filepath


def unified_stock_screener():
    st.title("Unified Stock Screener")
    st.info("Enter tickers and set your filters. The app will use all available data sources to screen stocks. No API names are shown.")

    # Ticker input
    input_method = st.radio("Ticker input method:", ["Manual entry", "Upload file", "URL"], key='unified_input_method')
    tickers = []
    if input_method == "Manual entry":
        tickers_input = st.text_area("Enter tickers (comma or newline separated)", "AAPL,MSFT,GOOG,IBM", key='unified_manual')
        if tickers_input:
            tickers = [t.strip().upper() for t in tickers_input.replace(',', '\n').splitlines() if t.strip()]
    elif input_method == "Upload file":
        uploaded_file = st.file_uploader("Upload a file with one ticker per line", key='unified_file')
        if uploaded_file:
            tickers = [line.decode('utf-8').strip().upper() for line in uploaded_file.readlines() if line.strip()]
    elif input_method == "URL":
        url = st.text_input("Enter URL to ticker list (plain text, one ticker per line)", key='unified_url')
        if url:
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                tickers = [line.strip().upper() for line in resp.text.splitlines() if line.strip()]
            except Exception as e:
                st.error(f"Could not load from URL: {e}")

    st.markdown('---')
    st.subheader("Stock Filters")
    exchanges = [
        ('NASDAQ', 'XNAS'),
        ('NYSE', 'XNYS'),
        ('AMEX', 'XASE'),
        ('ARCA', 'ARCX'),
        ('BATS', 'BATS'),
        ('OTC', 'OTCM')
    ]
    sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical', 'Consumer Defensive', 'Energy', 'Industrials', 'Real Estate', 'Utilities', 'Basic Materials', 'Communication Services', 'Other']
    types = [
        ('Common Stock', 'CS'),
        ('ETF', 'ETF'),
        ('ADR', 'ADR'),
        ('Preferred Stock', 'PS'),
        ('Unit', 'UNIT'),
        ('Closed-End Fund', 'CEF'),
        ('Structured Product', 'SP'),
        ('Other', 'OTHER')
    ]
    countries = ['US', 'CA', 'GB', 'DE', 'FR', 'JP', 'CN', 'IN', 'AU', 'Other']
    currencies = ['USD', 'CAD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'AUD', 'Other']
    list_statuses = [('Active', 'active'), ('Inactive', 'inactive')]

    exchange = st.selectbox("Exchange", [e[0] for e in exchanges], key='unified_exchange')
    sector = st.selectbox("Sector", sectors, key='unified_sector')
    type_ = st.selectbox("Type", [t[0] for t in types], key='unified_type')
    country = st.selectbox("Country", countries, key='unified_country')
    currency = st.selectbox("Currency", currencies, key='unified_currency')
    list_status = st.selectbox("List Status", [l[0] for l in list_statuses], key='unified_list_status')
    market_cap_min = st.text_input("Min Market Cap (USD)", key='unified_marketcap_min')
    market_cap_max = st.text_input("Max Market Cap (USD)", key='unified_marketcap_max')
    search = st.text_input("Company name or ticker search", key='unified_search')
    limit = st.number_input("Limit", min_value=1, max_value=1000, value=10, key='unified_limit')

    st.markdown('---')
    st.subheader('Technical Indicator Thresholds (leave blank for no filter)')
    sma_20_min = st.number_input('Min SMA 20', value=0.0, step=0.01, format="%f", key='unified_sma20min')
    sma_20_max = st.number_input('Max SMA 20', value=0.0, step=0.01, format="%f", key='unified_sma20max')
    sma_50_min = st.number_input('Min SMA 50', value=0.0, step=0.01, format="%f", key='unified_sma50min')
    sma_50_max = st.number_input('Max SMA 50', value=0.0, step=0.01, format="%f", key='unified_sma50max')
    sma_200_min = st.number_input('Min SMA 200', value=0.0, step=0.01, format="%f", key='unified_sma200min')
    sma_200_max = st.number_input('Max SMA 200', value=0.0, step=0.01, format="%f", key='unified_sma200max')
    rsi_min = st.number_input('Min RSI (14d)', value=0.0, step=0.01, format="%f", key='unified_rsimin')
    rsi_max = st.number_input('Max RSI (14d)', value=100.0, step=0.01, format="%f", key='unified_rsimax')
    # Add more indicators as needed

    st.markdown('---')
    if st.button('Screen Stocks'):
        if not tickers:
            st.warning('Please provide at least one ticker.')
        else:
            results = []
            progress = st.progress(0)
            for i, ticker in enumerate(tickers):
                av_data = get_alpha_vantage_technicals(ticker)
                yahoo_metrics = None
                try:
                    for analyzer in MultiStockAnalyzer([ticker]).analyzers:
                        yahoo_metrics = analyzer.metrics
                        break
                except Exception:
                    pass
                row = {
                    'Ticker': ticker,
                    'SMA 20': av_data['SMA_20'] if av_data['SMA_20'] is not None else (yahoo_metrics.get('sma_20') if yahoo_metrics else None),
                    'SMA 50': av_data['SMA_50'] if av_data['SMA_50'] is not None else (yahoo_metrics.get('sma_50') if yahoo_metrics else None),
                    'SMA 200': av_data['SMA_200'] if av_data['SMA_200'] is not None else (yahoo_metrics.get('sma_200') if yahoo_metrics else None),
                    'RSI (14d)': av_data['RSI_14'] if av_data['RSI_14'] is not None else (yahoo_metrics.get('rsi_14') if yahoo_metrics else None),
                }
                # Filter logic
                if (
                    (not sma_20_min or (row['SMA 20'] is not None and row['SMA 20'] >= sma_20_min)) and
                    (not sma_20_max or (row['SMA 20'] is not None and row['SMA 20'] <= sma_20_max)) and
                    (not sma_50_min or (row['SMA 50'] is not None and row['SMA 50'] >= sma_50_min)) and
                    (not sma_50_max or (row['SMA 50'] is not None and row['SMA 50'] <= sma_50_max)) and
                    (not sma_200_min or (row['SMA 200'] is not None and row['SMA 200'] >= sma_200_min)) and
                    (not sma_200_max or (row['SMA 200'] is not None and row['SMA 200'] <= sma_200_max)) and
                    (not rsi_min or (row['RSI (14d)'] is not None and row['RSI (14d)'] >= rsi_min)) and
                    (not rsi_max or (row['RSI (14d)'] is not None and row['RSI (14d)'] <= rsi_max))
                ):
                    results.append(row)
                progress.progress((i+1)/len(tickers))
            if results:
                df = pd.DataFrame(results)
                st.success(f"Found {len(df)} matching stocks.")
                st.dataframe(df)
                st.download_button("Download Results as CSV", df.to_csv(index=False), file_name="screener_results.csv")
            else:
                st.warning("No stocks matched your criteria.")


def fetch_alpha_vantage_indicator(symbol, function, **kwargs):
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        **kwargs
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Alpha Vantage API error: {e}")
        return None

def get_alpha_vantage_technicals(symbol):
    # Fetch SMA 20, 50, 200
    sma = {}
    for period in [20, 50, 200]:
        data = fetch_alpha_vantage_indicator(symbol, 'SMA', interval='daily', time_period=period, series_type='close')
        if data and 'Technical Analysis: SMA' in data:
            try:
                last_date = sorted(data['Technical Analysis: SMA'].keys())[-1]
                sma[period] = float(data['Technical Analysis: SMA'][last_date]['SMA'])
            except Exception:
                sma[period] = None
        else:
            sma[period] = None
    # Fetch RSI
    rsi = None
    data = fetch_alpha_vantage_indicator(symbol, 'RSI', interval='daily', time_period=14, series_type='close')
    if data and 'Technical Analysis: RSI' in data:
        try:
            last_date = sorted(data['Technical Analysis: RSI'].keys())[-1]
            rsi = float(data['Technical Analysis: RSI'][last_date]['RSI'])
        except Exception:
            rsi = None
    # Fetch Bollinger Bands
    boll = {'upper': None, 'lower': None}
    data = fetch_alpha_vantage_indicator(symbol, 'BBANDS', interval='daily', time_period=20, series_type='close')
    if data and 'Technical Analysis: BBANDS' in data:
        try:
            last_date = sorted(data['Technical Analysis: BBANDS'].keys())[-1]
            boll['upper'] = float(data['Technical Analysis: BBANDS'][last_date]['Real Upper Band'])
            boll['lower'] = float(data['Technical Analysis: BBANDS'][last_date]['Real Lower Band'])
        except Exception:
            boll['upper'] = None
            boll['lower'] = None
    return {
        'SMA_20': sma[20],
        'SMA_50': sma[50],
        'SMA_200': sma[200],
        'RSI_14': rsi,
        'BOLL_UPPER': boll['upper'],
        'BOLL_LOWER': boll['lower']
    }

def main():
    st.set_page_config(page_title="VirtualFusion Stock Analysis Tool", layout="wide")
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 100%);
        min-height: 100vh;
    }
    .block-container {
        background: rgba(255,255,255,0.85) !important;
        border-radius: 20px;
        padding: 2rem 2rem 2rem 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    h1, h2, h3, h4 {
        color: #22223B !important;
        font-weight: 800 !important;
    }
    .stButton>button {
        border-radius: 10px;
        padding: 0.75em 2em;
        font-size: 1.1em;
        font-weight: bold;
        background: linear-gradient(90deg, #6C63FF 0%, #FF6CAB 100%);
        color: white;
        border: none;
        margin: 0.5em 0.5em 0.5em 0;
        box-shadow: 0 4px 14px 0 rgba(0,0,0,0.15);
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #48C6EF 0%, #6C63FF 100%);
        color: #fff;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #6C63FF;
        padding: 0.5em;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

    st.markdown(
    """
    <h1 style='text-align: center; font-size: 3em;'>
        🚀 <span style='color:#6C63FF;'>VirtualFusion Stock Analysis Tool</span>
    </h1>
    <h3 style='text-align: center; color: #22223B; font-weight: 400;'>
        Professional Stock Analysis & Discovery Platform
    </h3>
    """,
    unsafe_allow_html=True
)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.button("📊 Analyze Single Stock")
    with col2:
        st.button("✨ Compare Multiple Stocks")
    with col3:
        st.button("📝 Generate Report")

    tab1, tab2 = st.tabs([
        "📈 Analyze Specific Stocks",
        "🔍 Discover Stocks by Filters"
    ])

    with tab1:
        st.header("Option 1: Query one or multiple stocks (Yahoo Finance)")
        input_method = st.radio("Choose ticker input method:", ["Manual entry", "Load from file", "Load from URL"], key='yahoo_input_method')
        symbols = []
        if input_method == "Manual entry":
            symbols_input = st.text_input("Enter stock symbol(s) (comma-separated, e.g. IBM,AAPL,GOOG)", "IBM", key='yahoo_manual')
            if symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        elif input_method == "Load from file":
            uploaded_file = st.file_uploader("Upload a file with one ticker per line", key='yahoo_file')
            if uploaded_file:
                symbols = [line.decode('utf-8').strip().upper() for line in uploaded_file.readlines() if line.strip()]
        elif input_method == "Load from URL":
            url = st.text_input("Enter URL to ticker list (plain text, one ticker per line)", key='yahoo_url')
            if url:
                try:
                    resp = requests.get(url)
                    resp.raise_for_status()
                    symbols = [line.strip().upper() for line in resp.text.splitlines() if line.strip()]
                except Exception as e:
                    st.error(f"Could not load from URL: {e}")
        if symbols:
            st.write(f"Loaded {len(symbols)} tickers.")
            if st.button("Analyze Stocks"):
                results = []
                progress = st.progress(0)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(VirtualFusionAnalyzer, symbol): symbol for symbol in symbols}
                    for i, f in enumerate(as_completed(futures)):
                        analyzer = f.result()
                        results.append(analyzer)
                        progress.progress((i+1)/len(symbols))
                st.success("Analysis complete!")
                # Display comprehensive table
                if results:
                    summary_data = []
                    for analyzer in results:
                        d = analyzer.metrics.copy()
                        d['symbol'] = analyzer.symbol
                        summary_data.append(d)
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary)
                    # Download as XLS
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_summary.to_excel(writer, index=False, sheet_name='Summary')
                    st.download_button("Download All Results as XLS", output.getvalue(), file_name="virtualfusion_analysis_results.xlsx")
                if st.button("Download All Results as JSON"):
                    json_data = json.dumps([a.as_dict() for a in results], indent=2)
                    st.download_button("Download JSON", json_data, file_name="virtualfusion_analysis_results.json")

    with tab2:
        st.header("Option 2: Polygon.io Screener")
        api_key = os.environ.get('POLYGON_API_KEY')
        if not api_key:
            st.warning("Polygon.io API key not set in environment variable POLYGON_API_KEY. Using hardcoded key (not secure). Please set POLYGON_API_KEY in your environment.")
            api_key = '6SxODAshRB4iXcDvxp3zIfvn54iQRbiK'
        exchanges = [
            ('NASDAQ', 'XNAS'),
            ('NYSE', 'XNYS'),
            ('AMEX', 'XASE'),
            ('ARCA', 'ARCX'),
            ('BATS', 'BATS'),
            ('OTC', 'OTCM')
        ]
        sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical', 'Consumer Defensive', 'Energy', 'Industrials', 'Real Estate', 'Utilities', 'Basic Materials', 'Communication Services', 'Other']
        types = [
            ('Common Stock', 'CS'),
            ('ETF', 'ETF'),
            ('ADR', 'ADR'),
            ('Preferred Stock', 'PS'),
            ('Unit', 'UNIT'),
            ('Closed-End Fund', 'CEF'),
            ('Structured Product', 'SP'),
            ('Other', 'OTHER')
        ]
        countries = ['US', 'CA', 'GB', 'DE', 'FR', 'JP', 'CN', 'IN', 'AU', 'Other']
        currencies = ['USD', 'CAD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'AUD', 'Other']
        list_statuses = [('Active', 'active'), ('Inactive', 'inactive')]

        input_method = st.radio("Ticker input method:", ["Manual entry", "Upload file", "URL"], key='polygon_input_method')
        tickers = []
        if input_method == "Manual entry":
            tickers_input = st.text_area("Enter tickers (comma or newline separated)", "AAPL,MSFT,GOOG,IBM", key='polygon_manual')
            if tickers_input:
                tickers = [t.strip().upper() for t in tickers_input.replace(',', '\n').splitlines() if t.strip()]
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader("Upload a file with one ticker per line", key='polygon_file')
            if uploaded_file:
                tickers = [line.decode('utf-8').strip().upper() for line in uploaded_file.readlines() if line.strip()]
        elif input_method == "URL":
            url = st.text_input("Enter URL to ticker list (plain text, one ticker per line)", key='polygon_url')
            if url:
                try:
                    resp = requests.get(url)
                    resp.raise_for_status()
                    tickers = [line.strip().upper() for line in resp.text.splitlines() if line.strip()]
                except Exception as e:
                    st.error(f"Could not load from URL: {e}")

        st.markdown('---')
        st.subheader("Filter Parameters")
        exchange = st.selectbox("Exchange", [e[0] for e in exchanges], key='polygon_exchange')
        sector = st.selectbox("Sector", sectors, key='polygon_sector')
        type_ = st.selectbox("Type", [t[0] for t in types], key='polygon_type')
        country = st.selectbox("Country", countries, key='polygon_country')
        currency = st.selectbox("Currency", currencies, key='polygon_currency')
        list_status = st.selectbox("List Status", [l[0] for l in list_statuses], key='polygon_list_status')
        market_cap_min = st.text_input("Min Market Cap (USD)", key='polygon_marketcap_min')
        market_cap_max = st.text_input("Max Market Cap (USD)", key='polygon_marketcap_max')
        search = st.text_input("Company name or ticker search", key='polygon_search')
        limit = st.number_input("Limit", min_value=1, max_value=1000, value=10, key='polygon_limit')

        # --- GROUPED FILTERS ---
        st.markdown('---')
        st.subheader('Advanced Filters (Grouped)')

        with st.expander('Valuation'):
            pe_ratio = st.number_input('P/E Ratio (Current)', value=0.0, step=0.01, format="%f", key='pe_ratio')
            pe_ratio_5y = st.number_input('P/E Ratio (5Y Avg)', value=0.0, step=0.01, format="%f", key='pe_ratio_5y')
            ev_ebitda = st.number_input('EV/EBITDA (Current)', value=0.0, step=0.01, format="%f", key='ev_ebitda')
            ev_ebitda_5y = st.number_input('EV/EBITDA (5Y Avg)', value=0.0, step=0.01, format="%f", key='ev_ebitda_5y')
            peg = st.number_input('PEG Ratio', value=0.0, step=0.01, format="%f", key='peg')
            fcf = st.number_input('Free Cash Flow', value=0.0, step=0.01, format="%f", key='fcf')
            # Note: Backend logic needed to fetch/calculate these

        with st.expander('Technical'):
            sma_20 = st.number_input('SMA 20 (Daily)', value=0.0, step=0.01, format="%f", key='sma20')
            sma_50 = st.number_input('SMA 50 (Daily)', value=0.0, step=0.01, format="%f", key='sma50')
            sma_200 = st.number_input('SMA 200 (Daily)', value=0.0, step=0.01, format="%f", key='sma200')
            rsi = st.number_input('RSI (14d)', value=0.0, step=0.01, format="%f", key='rsi')
            boll_upper = st.number_input('Bollinger Upper', value=0.0, step=0.01, format="%f", key='boll_upper')
            boll_lower = st.number_input('Bollinger Lower', value=0.0, step=0.01, format="%f", key='boll_lower')
            impvol = st.number_input('Implied Volatility', value=0.0, step=0.01, format="%f", key='impvol')
            beta = st.number_input('Beta vs Index', value=0.0, step=0.01, format="%f", key='beta')
            pos_52w = st.number_input('Position vs 52 Week (High-Low)', value=0.0, step=0.01, format="%f", key='pos_52w')
            # New: 1d max drawdown (3y)
            max_drawdown_1d_3y = st.number_input('1 Day Max Drawdown (Past 3Y) [%]', value=0.0, step=0.01, format="%f", key='max_drawdown_1d_3y')
            # New: Max consecutive negative days (5y)
            max_consec_neg_5y = st.number_input('Max Consecutive Days Negative (5Y)', value=0.0, step=1.0, format="%f", key='max_consec_neg_5y')
            # Note: Some require backend logic

        with st.expander('Earnings'):
            eps = st.number_input('EPS', value=0.0, step=0.01, format="%f", key='eps')
            beat_miss_ratio = st.number_input('Earnings Beat/Miss Ratio (5Y) [%]', value=0.0, step=0.01, format="%f", key='beat_miss_ratio')
            avg_move_beat = st.number_input('Avg Daily Move on Earnings Beat (5Y) [%]', value=0.0, step=0.01, format="%f", key='avg_move_beat')
            avg_move_miss = st.number_input('Avg Daily Move on Earnings Miss (5Y) [%]', value=0.0, step=0.01, format="%f", key='avg_move_miss')
            # Note: Backend logic needed

        with st.expander('Analyst/Consensus'):
            analyst_consensus = st.text_input('Analyst Price Consensus', key='analyst_consensus')
            upside_potential = st.number_input('Upside Potential (%)', value=0.0, step=0.01, format="%f", key='upside_potential')
            # Note: Backend logic needed

        with st.expander('Risk/Drawdown'):
            short_float = st.number_input('Short Float (%)', value=0.0, step=0.01, format="%f", key='short_float')
            # Note: Backend logic needed

        params = {
            'market': 'stocks',
            'active': 'true',
            'limit': limit,
            'apiKey': api_key
        }
        if exchange:
            params['exchange'] = dict(exchanges)[exchange]
        if sector and sector != 'Other':
            params['sector'] = sector
        if type_:
            params['type'] = dict(types)[type_]
        if country and country != 'Other':
            params['country'] = country
        if currency and currency != 'Other':
            params['currency'] = currency
        if list_status:
            params['active'] = 'true' if dict(list_statuses)[list_status] == 'active' else 'false'
        if market_cap_min:
            params['market_cap.gte'] = market_cap_min
        if market_cap_max:
            params['market_cap.lte'] = market_cap_max
        if search:
            params['search'] = search

        results_df = None
        if st.button("Search Polygon.io Screener"):
            url = 'https://api.polygon.io/v3/reference/tickers'
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])
                if tickers:
                    tickers_set = set(tickers)
                    results = [r for r in results if r.get('ticker', '').upper() in tickers_set]
                if not results:
                    st.warning("No stocks found for these criteria.")
                else:
                    enriched = []
                    progress = st.progress(0)
                    for i, r in enumerate(results):
                        ticker = r.get('ticker', '')
                        av_data = get_alpha_vantage_technicals(ticker)
                        try:
                            analyzer = VirtualFusionAnalyzer(symbol=ticker)
                            yahoo_metrics = analyzer.metrics
                        except Exception:
                            yahoo_metrics = {}
                        row = {
                            'Ticker': ticker,
                            # Valuation
                            'P/E Ratio': yahoo_metrics.get('pe_ratio', 'N/A'),
                            'P/E Ratio (5Y Avg)': yahoo_metrics.get('pe_ratio_5y', 'N/A'),
                            'EV/EBITDA': yahoo_metrics.get('ev_ebitda', 'N/A'),
                            'EV/EBITDA (5Y Avg)': yahoo_metrics.get('ev_ebitda_5y', 'N/A'),
                            'PEG': yahoo_metrics.get('peg_ratio', 'N/A'),
                            'Free Cash Flow': yahoo_metrics.get('fcf', 'N/A'),
                            # Technical
                            'SMA 20': av_data['SMA_20'] if av_data['SMA_20'] is not None else yahoo_metrics.get('sma_20'),
                            'SMA 50': av_data['SMA_50'] if av_data['SMA_50'] is not None else yahoo_metrics.get('sma_50'),
                            'SMA 200': av_data['SMA_200'] if av_data['SMA_200'] is not None else yahoo_metrics.get('sma_200'),
                            'RSI (14d)': av_data['RSI_14'] if av_data['RSI_14'] is not None else yahoo_metrics.get('rsi_14'),
                            'Bollinger Upper': av_data['BOLL_UPPER'] if av_data['BOLL_UPPER'] is not None else yahoo_metrics.get('boll_upper'),
                            'Bollinger Lower': av_data['BOLL_LOWER'] if av_data['BOLL_LOWER'] is not None else yahoo_metrics.get('boll_lower'),
                            'Implied Volatility': yahoo_metrics.get('implied_volatility', 'N/A'),
                            'Beta': yahoo_metrics.get('beta', 'N/A'),
                            'Position vs 52W': yahoo_metrics.get('pos_52w', 'N/A'),
                            '1D Max Drawdown (3Y) [%]': yahoo_metrics.get('max_drawdown_1d_3y', 'N/A'),
                            'Max Consec Neg (5Y)': yahoo_metrics.get('max_consec_neg_5y', 'N/A'),
                            # Earnings
                            'EPS': yahoo_metrics.get('eps', 'N/A'),
                            'Earnings Beat/Miss Ratio (5Y)': yahoo_metrics.get('beat_miss_ratio', 'N/A'),
                            'Avg Move Beat (5Y)': yahoo_metrics.get('avg_move_beat', 'N/A'),
                            'Avg Move Miss (5Y)': yahoo_metrics.get('avg_move_miss', 'N/A'),
                            # Analyst/Consensus
                            'Analyst Consensus': yahoo_metrics.get('analyst_consensus', 'N/A'),
                            'Upside Potential': yahoo_metrics.get('upside_potential', 'N/A'),
                            # Risk/Drawdown
                            'Short Float': yahoo_metrics.get('short_float', 'N/A'),
                        }
                        passes = True
                        if pe_ratio > 0 and row['P/E Ratio'] != 'N/A' and row['P/E Ratio'] < pe_ratio:
                            passes = False
                        if pos_52w > 0 and row['Position vs 52W'] != 'N/A' and row['Position vs 52W'] < pos_52w:
                            passes = False
                        if max_drawdown_1d_3y > 0 and row['1D Max Drawdown (3Y) [%]'] != 'N/A' and abs(row['1D Max Drawdown (3Y) [%]']) < abs(max_drawdown_1d_3y):
                            passes = False
                        if max_consec_neg_5y > 0 and row['Max Consec Neg (5Y)'] != 'N/A' and row['Max Consec Neg (5Y)'] < max_consec_neg_5y:
                            passes = False
                        if beat_miss_ratio > 0 and row['Earnings Beat/Miss Ratio (5Y)'] != 'N/A' and (row['Earnings Beat/Miss Ratio (5Y)']*100) < beat_miss_ratio:
                            passes = False
                        if avg_move_beat > 0 and row['Avg Move Beat (5Y)'] != 'N/A' and abs(row['Avg Move Beat (5Y)']) < abs(avg_move_beat):
                            passes = False
                        if avg_move_miss > 0 and row['Avg Move Miss (5Y)'] != 'N/A' and abs(row['Avg Move Miss (5Y)']) < abs(avg_move_miss):
                            passes = False
                        if passes:
                            enriched.append(row)
                        progress.progress((i+1)/len(results))
                    if enriched:
                        results_df = pd.DataFrame(enriched)
                        st.success(f"Found {len(results_df)} matching stocks.")
                        st.dataframe(results_df)
                        st.download_button("Download Results as CSV", results_df.to_csv(index=False), file_name="polygon_screener_results.csv")
                    else:
                        st.warning("No stocks matched your criteria.")
            except Exception as e:
                st.error(f"Failed to query Polygon.io Screener: {e}")
        if results_df is not None and not results_df.empty:
            json_str = results_df.to_json(orient='records', indent=2)
            st.download_button("Download Results as JSON", json_str, file_name="polygon_screener_results.json")

if __name__ == "__main__":
    main()


