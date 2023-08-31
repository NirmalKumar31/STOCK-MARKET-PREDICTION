# Import required libraries
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



# Define constants
START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set the title and page-wide styling
st.set_page_config(
    page_title="Stock Forecast App",
    page_icon="💹",
    layout="wide"
)

st.markdown('<p style="text-align: right;">Created by Nirmalkumar T K</p>', unsafe_allow_html=True)



# Custom CSS to style the app
st.markdown("""
<style>
body {
    background-color: #007BFF;  /* Change to the blue color you want */
    color: #ffffff;  /* Text color for better contrast */
    font-family: 'Arial', sans-serif;
}
.stButton>button {
    background-color: #ffffff;  /* White button background */
    color: #007BFF;  /* Blue button text color */
    font-weight: bold;
}
.stTextArea>div>div {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.stNumberInput>div {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.stPlotlyChart {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)
# Main app content
st.title('Stock Forecast App')

# Allow user to enter stock names
stock_names = st.text_input("Enter stock names (comma-separated)", "GOOG, AAPL, MSFT, GME")
stocks = [s.strip() for s in stock_names.split(',')]

# Styling for the slider and section headers
st.write("")
st.write("")
st.markdown("<h3 style='color:#007BFF;'>Configure Forecast</h3>", unsafe_allow_html=True)
n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

# Function to load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load and display data for each selected stock
for selected_stock in stocks:
    data_load_state = st.text(f'Loading data for {selected_stock}...')
    data = load_data(selected_stock)
    data_load_state.text(f'Loading data for {selected_stock}... done!')
    
    st.write("")
    st.markdown(f"<h2 style='color:#007BFF;'>{selected_stock} Data</h2>", unsafe_allow_html=True)
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text=f'Time Series data for {selected_stock} with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.write("")
    st.markdown(f"<h2 style='color:#007BFF;'>{selected_stock} Forecast</h2>", unsafe_allow_html=True)
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years for {selected_stock}')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.write(f"Forecast components for {selected_stock}")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
