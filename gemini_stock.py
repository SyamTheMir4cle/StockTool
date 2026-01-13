from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Input, Static, DataTable, TabbedContent, TabPane
from textual import on
import plotext as plt
import matplotlib.pyplot as mplt
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from datetime import datetime
import re

class SyamStock(App):
    """
    Stock CLI (Final Version).
    Features: Command Bar, AI Insight, Terminal Plots, and Full GUI with RSI.
    """

    CSS = """
    Screen {
        layout: vertical;
        background: #0f111a;  /* Deep Space Blue */
    }

    /* --- Command Bar (Bottom) --- */
    #cmd-container {
        dock: bottom;
        height: auto;
        padding: 1 2;
        background: #1a1d29;
        border-top: solid #00d2ff; 
    }

    Input {
        width: 100%;
        background: #0f111a;
        border: none;
        color: #00d2ff;
    }
    Input:focus {
        border: wide #00d2ff;
    }

    /* --- Output Area --- */
    #main-content {
        height: 1fr;
        padding: 1;
    }

    TabbedContent {
        height: 100%;
    }
    
    #plot-area {
        height: 100%;
        min-height: 25;
        border: solid #333;
        background: #000;
        color: #eee;
    }

    #insight-box {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: #1e2230;
        color: #e1e1e1;
        border-left: wide #00d2ff;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("m", "show_gui", "Matplotlib GUI"),
        ("s", "save_csv", "Save CSV"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="main-content"):
            yield Static("Welcome. Type a ticker below (e.g., 'BTC' or 'XAUUSD').", id="insight-box")
            
            with TabbedContent(id="tabs"):
                with TabPane("Visual Analysis", id="tab-chart"):
                    yield Static("", id="plot-area")
                
                with TabPane("Raw Data", id="tab-data"):
                    yield DataTable(zebra_stripes=True)

        with Container(id="cmd-container"):
            yield Input(placeholder="> Ask Stock tool (e.g., 'NVDA', 'GOLD 1y', 'BTC 2023-01-01')", id="cmd-input")

        yield Footer()

    def on_mount(self):
        self.query_one("#cmd-input").focus()

    @on(Input.Submitted)
    def handle_command(self, event: Input.Submitted):
        command = event.value.strip()
        if not command: return
        self.process_command(command)
        event.input.value = ""

    def process_command(self, command):
        parts = command.split()
        raw_ticker = parts[0].upper()
        
        start_date = None
        end_date = None
        
        # Regex to find dates like YYYY-MM-DD
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        dates = re.findall(date_pattern, command)
        
        if len(dates) >= 1: start_date = dates[0]
        if len(dates) >= 2: end_date = dates[1]

        # Smart Mapping
        symbol_map = {
            "XAUUSD": "GC=F", "GOLD": "GC=F", "OIL": "CL=F", 
            "BTC": "BTC-USD", "ETH": "ETH-USD", "EURUSD": "EURUSD=X"
        }
        fetch_ticker = symbol_map.get(raw_ticker, raw_ticker)

        self.query_one("#insight-box").update(f"Scanning market data for **{fetch_ticker}**...")
        
        try:
            df = self.fetch_data(fetch_ticker, start_date, end_date)
            if df is None:
                self.query_one("#insight-box").update(f"❌ Could not find data for {raw_ticker}.")
                return

            self.current_df = df
            self.current_ticker = raw_ticker

            insight = self.generate_insight(df, raw_ticker)
            self.query_one("#insight-box").update(insight)

            self.query_one("#tabs").active = "tab-chart"
            self.update_terminal_plot(df, raw_ticker)
            self.update_table(df)

        except Exception as e:
            self.query_one("#insight-box").update(f"❌ System Error: {str(e)}")

    def fetch_data(self, ticker, start, end):
        if start:
            if not end: end = datetime.now().strftime('%Y-%m-%d')
            df = yf.Ticker(ticker).history(start=start, end=end)
        else:
            df = yf.Ticker(ticker).history(period="1y")

        if df.empty: return None

        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        df = df.round(2)
        df.reset_index(inplace=True)
        # Helper column for terminal plotting
        df['Date_Str'] = df['Date'].dt.strftime('%d/%m/%Y')
        return df

    def generate_insight(self, df, ticker):
        last_price = df['Close'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        last_sma = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else last_price

        trend = "BULLISH 🟢" if last_price > last_sma else "BEARISH 🔴"
        
        rsi_status = "NEUTRAL"
        if last_rsi > 70: rsi_status = "OVERBOUGHT (High Risk) ⚠️"
        elif last_rsi < 30: rsi_status = "OVERSOLD (Potential Buy) 💎"

        return (
            f"**ANALYSIS REPORT: {ticker}**\n"
            f"• Price: ${last_price:,.2f} | Trend: {trend}\n"
            f"• RSI: {last_rsi:.1f} - {rsi_status}\n"
            f"• Support (SMA50): ${last_sma:,.2f}\n"
            f"• *Press 'm' for detailed GUI charts.*"
        )

    def update_terminal_plot(self, df, ticker):
        plot_area = self.query_one("#plot-area", Static)
        w, h = plot_area.content_size
        if w == 0: w, h = 100, 30

        plt.clear_figure()
        plt.subplots(2, 1)

        # Price
        plt.subplot(1, 1)
        plt.theme('dark')
        plt.plot(df['Close'].tolist(), label="Price", color="white")
        plt.plot(df['SMA_50'].tolist(), label="SMA 50", color="blue")
        plt.title(f"{ticker} Market View")
        plt.grid(True, True)

        # RSI
        plt.subplot(2, 1)
        plt.plot(df['RSI'].tolist(), label="RSI", color="purple")
        plt.hline(70, color="red")
        plt.hline(30, color="green")
        plt.ylim(0, 100)
        plt.grid(True, True)

        plt.plotsize(w, h)
        plot_area.update(plt.build())

    def update_table(self, df):
        table = self.query_one(DataTable)
        table.clear(columns=True)
        columns = ["Date_Str", "Open", "Close", "SMA_50", "RSI"]
        table.add_columns(*columns)
        table.add_rows(df[columns].iloc[::-1].head(100).values.tolist())

    def action_show_gui(self):
        """Triggered by pressing 'm'. Shows Price AND RSI."""
        if not hasattr(self, 'current_df'): return
        
        df = self.current_df
        ticker = self.current_ticker
        
        # Create a window with 2 rows (Price Top, RSI Bottom)
        fig, (ax1, ax2) = mplt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # --- Top Panel: Price & SMA ---
        ax1.plot(df['Date'], df['Close'], label='Price', color='black', linewidth=1.5)
        ax1.plot(df['Date'], df['SMA_50'], label='SMA 50', color='blue', linestyle='--')
        ax1.set_title(f"{ticker} Technical Analysis")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # --- Bottom Panel: RSI ---
        ax2.plot(df['Date'], df['RSI'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax2.fill_between(df['Date'], 70, 30, color='gray', alpha=0.1)
        
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        mplt.tight_layout()
        mplt.show()

    def action_save_csv(self):
        """Triggered by pressing 's'"""
        if hasattr(self, 'current_df'):
            fn = f"{self.current_ticker}_gemini.csv"
            self.current_df.to_csv(fn)
            self.query_one("#insight-box").update(f"✅ Data saved to {fn}")

if __name__ == "__main__":
    app = SyamStock()
    app.run()