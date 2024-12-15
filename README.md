# Portfolio Analysis App

## Overview
- This web application allows users to analyze and manage a portfolio of stocks. The app leverages data from the S&P 500 index and individual S&P 500 stocks to compute portfolio performance, including returns, volatility, Sharpe ratio, and correlations. The user can select stocks, assign weights to each, and track the portfolio's historical performance against the benchmark index.

## Features
- Stock Selection: Users can choose from a list of S&P 500 stocks to add to their portfolio.
- Portfolio Weighting: Allocate weights to selected stocks, with the option to adjust individual stock weights.
- Performance Tracking: View the historical performance of the selected portfolio and compare it against the S&P 500 index.
- Portfolio Metrics: The app calculates key performance metrics such as:
  - Annual Return
  - Annual Volatility
  - Sharpe Ratio
- Correlation Matrix: Visualizes the correlation between the selected stocks.

## Data
The app uses historical data of the S&P 500 companies, index, and individual stock prices. 
The data is sourced from the following CSV files:
- sp500_stocks.csv: Historical stock prices of S&P 500 companies.
- sp500_index.csv: Historical data of the S&P 500 index.
- datasp500_companies.csv: A list of S&P 500 companies, including ticker symbols and industries.

## Data Sources:
- The S&P 500 stock data and index data are sourced from publicly available datasets and can be updated periodically.

## Requirements
- Before running the app, ensure that you have the necessary dependencies installed. The required libraries are listed in the requirements.txt file. You can install them using pip.

pip install -r requirements.txt

## Required Libraries:
- streamlit - For creating the interactive web app.
- pandas - For handling data manipulation.
- numpy - For numerical computations.
- plotly - For interactive visualizations.

## How to Run
- Clone this repository
- cd portfolio-analysis-app

## Install the required dependencies:
- pip install -r requirements.txt

### Start the Streamlit app:
- streamlit run app.py
- The app will open in your default web browser at http://localhost:8501.

## Usage
- Select Stocks for Portfolio: Use the sidebar to select the stocks you want to include in your portfolio. You can select multiple stocks.
- Adjust Weights: For each selected stock, adjust the weight using the sliders in the sidebar.
- View Portfolio Performance: The app will calculate and display the portfolio's historical performance, comparing it to the S&P 500 index.
- Analyze Portfolio Metrics: View important metrics like annual return, volatility, and Sharpe ratio.
- View Correlation Matrix: See how the selected stocks correlate with each other.

## Directory Structure
├── app.py                    # Main application code
├── datasp500_companies.csv   # List of S&P 500 companies and their details
├── sp500_index.csv           # S&P 500 index data
├── sp500_stocks.csv          # Historical data for S&P 500 stocks
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies

## Contributing
- Contributions are welcome! If you have any suggestions, bug fixes, or improvements, feel free to open an issue or submit a pull request.

## License
- This project is open-source and available under the MIT License.
