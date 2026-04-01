import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf 
from tensorflow.keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2026-03-26'


st.title('Stock Trend Prediction')

user_input =st.text_input('Enter Stock Ticker' , 'AAPL')
with st.spinner("Fetching Data..."):
    df = yf.download(user_input, start=start, end=end)
#Describing Data
st.subheader('Data from 2010-2026')
st.write(df.describe())



#visulizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing Price vs Time chart with 100MA & 200Ma')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)



#splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler

data = df[['Close']]

scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

data_training_array = scaler.transform(data_training)





#Load my model
Model = load_model('Keras_model.h5')

#Testing Part



past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = Model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]

y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test)


# ================= LIVE PRICE =================
df_live = yf.download(user_input, period="5y")
current_price = float(df_live['Close'].iloc[-1])

#Final graph
st.subheader('Predictions vs Original + Live')

fig2 = plt.figure(figsize=(12,6))

plt.plot(y_test, label='Original', linewidth=2)
plt.plot(y_predicted, label='Predicted', linewidth=2)

plt.axhline(y=current_price, linestyle='--', label='Live Price')

plt.grid(alpha=0.3)
plt.legend()
st.pyplot(fig2)





df_live = yf.download(user_input, period="5y")



st.line_chart(df['Close'])

if y_predicted[-1] > current_price:
    st.success("📈 Strong Uptrend Expected")
else:
    st.error("📉 Downtrend Expected")
    st.subheader("📍 Current Live Price")

st.metric(
    label="Stock Price",
    value=f"₹ {current_price:.2f}",
    delta=f"{(current_price - float(y_test[-1])):.2f}"
)

# last real data
plt.axhline(y=current_price, color='g', linestyle='--', label='Live Price')

current_price = df_live['Close'].iloc[-1]
current_date = df_live.index[-1]



current_price = float(df_live['Close'].iloc[-1])

st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.text_input("Enter Stock", "AAPL")
days = st.sidebar.slider("Future Days", 10, 100, 10)



st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


upper = y_predicted + 5
lower = y_predicted - 5

plt.fill_between(range(len(y_predicted.flatten())),
                 lower.flatten(),
                 upper.flatten(),
                 alpha=0.2)






# ================= FUTURE PREDICTION ================= #

st.subheader('🔮 Future Predictions')

# LIVE latest data (auto)
df_live = yf.download(user_input, period="5y")

data_live = df_live[['Close']]

# IMPORTANT: scaler 
input_data = scaler.transform(data)


# last 100 days input
time_step = 100
scaler.fit(data_training)
data_scaled = scaler.transform(data)

# ---------- NEXT 10 DAYS TABLE ---------- #
future_10 = []
last_data = data_scaled[-100:]
current_input = last_data.copy()
for i in range(10):
    current_input_reshaped = current_input.reshape(1, time_step, 1)
    pred = Model.predict(current_input_reshaped, verbose=0)
    
    future_10.append(pred[0][0])
    current_input = np.vstack((current_input[1:], pred))

# inverse scale
future_10 = np.array(future_10).reshape(-1,1)
future_10 = scaler.inverse_transform(future_10) # same scaling

# dates
from datetime import timedelta
last_date = df_live.index[-1]
dates_10 = [last_date + timedelta(days=i+1) for i in range(10)]

# table
df_10 = pd.DataFrame({
    "Date": dates_10,
    "Predicted Price": future_10.flatten()
})



st.subheader("📅 Next 10 Days Prediction (Table)")
st.write(df_10)

# ---------- NEXT 100 DAYS GRAPH ---------- #
future_100 = []
current_input = last_data.copy()

for i in range(100):
    current_input_reshaped = current_input.reshape(1, time_step, 1)
    pred = Model.predict(current_input_reshaped, verbose=0)
    
    future_100.append(pred[0][0])
    current_input = np.vstack((current_input[1:], pred))

# inverse scale
future_100 = np.array(future_100).reshape(-1,1)
future_100 = future_100 * scale_factor

# graph
st.subheader("📈 Next 100 Days Prediction (Graph)")
fig3 = plt.figure(figsize=(12,6))
plt.plot(future_100, 'g', label='Future 100 Days')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig3)