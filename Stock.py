import streamlit as st
import yfinance as yf  # Import yfinance directly in main.py
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for visualization

def get_stock_data(stock_ticker):
    """
    Fetches historical stock data for the given ticker symbol.
    """
    yfin = yf.Ticker(stock_ticker)
    hist = yfin.history(period="max")
    hist = hist[['Close']]
    hist.reset_index(inplace=True)
    hist.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    hist['ds'] = pd.to_datetime(hist['ds']).dt.tz_localize(None)  # Remove timezone
    return hist

def predict_future_price(hist, forecast_days):
    """
    Predicts future stock price using Prophet model.
    """
    m = Prophet()
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='weekly', period=7, fourier_order=3)
    m.fit(hist)  # Fit the model with historical data

    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    return m, forecast

def visualize_forecast_components(m, forecast, current_price, forecast_days):
    """
    Visualizes and displays forecast components.
    """
    # Plot current price and expected price after forecast days
    plt.figure(figsize=(12, 6))  # Adjust figure size for better display
    plt.plot(forecast['ds'], forecast['yhat'], color='red', linestyle='--', label='Forecasted Price')
    plt.axhline(y=current_price, color='blue', linestyle='-', label='Current Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Forecast')
    plt.legend()

    # Display the plot using Streamlit
    st.pyplot(plt.gcf())  # Display the generated figure

    # Calculate expected price after forecast days
    expected_price = forecast['yhat'].iloc[-1]

    # Calculate percentage appreciation
    percentage_appreciation = ((expected_price - current_price) / current_price) * 100

    # Display informative output using Streamlit elements
    st.write(f"Current Price: {current_price}")
    st.write(f"Expected Price after {forecast_days} days: {expected_price}")
    st.write(f"Percentage Appreciation expected: {percentage_appreciation:.2f}%")

    # Extract the date of the forecasted price
    forecast_date = forecast.iloc[-1]['ds']
    st.write(f"Date of Forecasted Price: {forecast_date}")

def main():
    # Create Streamlit header with CSS styling
    st.markdown("""
       <style>
           .header {
               color: #3498db;  /* Adjust header text color */
               text-align: center;
               font-size: 30px;  /* Adjust header font size */
               padding: 20px;  /* Adjust header padding */
           }
       </style>
       """, unsafe_allow_html=True)

    st.markdown('<h1 class="header">Stock Price Forecast App</h1>', unsafe_allow_html=True)  # Set the title

    # Create Streamlit header (choose a title you prefer)
    #st.title("Stock Price Forecast App")
    # Create Streamlit text input and selectbox for user interaction
    stock_ticker = st.text_input("Enter stock ticker symbol (e.g., INFY.NS): ")

    # Ensure forecast_days is an integer (avoid potential string conversion)
    selected_day_index = st.selectbox("Select the number of forecast days:", range(30, 366, 30))

    if st.button("Predict"):  # Button click triggers the prediction
        try:
            hist = get_stock_data(stock_ticker)
            current_price = hist['y'].iloc[-1]  # Get the current price from the last data point

            m, forecast = predict_future_price(hist, selected_day_index)

            # Visualize forecast components using Streamlit
            visualize_forecast_components(m, forecast, current_price, selected_day_index)

        except Exception as e:
            st.error(f"Error processing {stock_ticker}: {e}")
        # Add developer information and LinkedIn links (assuming you have their profiles)
        #st.write("Developed by:")
        #sachin_link = "[Sachin Shinkar](https://www.linkedin.com/in/sachinshinkar/)"  # Replace with Sachin's LinkedIn URL
        #st.write(sachin_link, unsafe_allow_html=True)  # Mark the HTML content as safe

        #vishal_link = "[Vishal Wani](https://www.linkedin.com/in/vishal-wani-b006111b0/)"  # Replace with Vishal's LinkedIn URL
        #st.write(vishal_link, unsafe_allow_html=True)
    st.write("---")

    st.write("Developed by: **[Sachin Shinkar](https://www.linkedin.com/in/sachinshinkar/)** and **[Vishal Wani](https://www.linkedin.com/in/vishal-wani-b006111b0/)**")

if __name__ == "__main__":
    main()
