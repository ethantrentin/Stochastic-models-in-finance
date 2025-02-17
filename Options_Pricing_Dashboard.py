import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

# First let's define a Black-Scholes function
def black_scholes_european(S,K,T,r,sigma, option_type="call"):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type == "put":
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Now, let's define a function to calculate all the Greeks of the option
def greeks_option(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) # Partial derivative with respect to underlying asset
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T)) # Second partial derivative with respect to underlying asset
    theta = -(sigma*S*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    vega = S*np.sqrt(T)*norm.pdf(d1)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2)
    return delta, gamma, theta, vega, rho

# We can now start to define our dashboard 
st.title("Options Pricing Dashboard")

st.sidebar.header("Option Calibration Parameters")
price_spot = st.sidebar.slider("Spot Price S (in $)", 50, 150, 100)
price_strike = st.sidebar.slider("Strike Price K (in $)", 50, 150, 100)
maturity = st.sidebar.slider("Time to maturity T (in years)", 0.1, 2.0, 1.0, 0.1)
risk_free_rate = st.sidebar.slider("Risk-free rate r ", 0.0, 0.1, 0.05, 0.01)
volatility = st.sidebar.slider("Volatility sigma ", 0.1, 1.0, 0.2, 0.01)
option_type = st.sidebar.radio("Option  Type", ["Call", "Put"])

# Calculate the option price 
option_price = black_scholes_european(price_spot, price_strike, maturity, risk_free_rate, volatility, option_type.lower())
st.write(f"### {option_type} Option Price : ${option_price:.2f}")

# Calculate the Greeks
delta, gamma, theta, vega, rho = greeks_option(price_spot, price_strike, maturity, risk_free_rate, volatility)
st.write("### Greeks")
st.write(f"**Delta**: {delta:.2f}")
st.write(f"**Gamma**: {gamma:.2f}")
st.write(f"**Vega**: {vega:.2f}")
st.write(f"**Theta**: {theta:.2f}")
st.write(f"**Rho**: {rho:.2f}")

# Volatility Smile 
st.write("### Volatility Smile")
prices_strike_linspace = np.linspace(50,150,100)
volatility_implied = [volatility for _ in prices_strike_linspace]

# Plot Smile
plt.figure(figsize=(10,6))
plt.plot(prices_strike_linspace, volatility_implied, label="Implied Volatility")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Volatility Smile")
plt.legend()
st.pyplot(plt)