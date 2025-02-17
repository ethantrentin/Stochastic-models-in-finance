import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import panel as pn

pn.extension() # This is needed to display the panel 

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

# We can now create widgets for user inputs
price_spot = pn.widgets.FloatSlider(name="Spot Price S (in $)", start=50, end=150, value=100, step=1)
price_strike = pn.widgets.FloatSlider(name="Strike Price K (in $)", start=50, end=150, value=100, step=1)
maturity = pn.widgets.FloatSlider(name="Time to maturity T (in years)", start=0.1, end=2.0, value=1.0, step=0.1)
risk_free_rate = pn.widgets.FloatSlider(name="Risk-free rate r ", start=0.0, end=0.1, value=0.05, step=0.01)
volatility = pn.widgets.FloatSlider(name="Volatility sigma ", start=0.0, end=1.0, value=0.2, step=0.01)
option_type = pn.widgets.RadioButtonGroup(name="Option  Type", options=["Call", "Put"], button_type="success")

# We are updating the dashboard 
@pn.depends(price_spot.param.value, price_strike.param.value, maturity.param.value, risk_free_rate.param.value, volatility.param.value, option_type.param.value)

def update(S, K, T, r, sigma, option_type):
    # Calculation of the option price
    option_price = black_scholes_european(S, K, T, r, sigma, option_type.lower())
    # Calculation of the Greeks
    delta, gamma, theta, vega, rho = greeks_option(S, K, T, r, sigma)
    # Volatility Smile 
    prices_strike_linspace = np.linspace(50,150,100)
    volatility_implied = [volatility for _ in prices_strike_linspace]
    # Plot the Volatility Smile
    plt.figure(figsize=(10,6))
    plt.plot(prices_strike_linspace, volatility_implied, label="Implied Volatility")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title("Volatility Smile")
    plt.legend()
    # Return the results and plot
    return pn.Column(f"### {option_type} Option Price : ${option_price:.2f}", f"**Delta**: {delta:.2f}", f"**Gamma**: {gamma:.2f}", f"**Vega**: {vega:.2f}", f"**Theta**: {theta:.2f}", f"**Rho**: {rho:.2f}", plt.gcf())

# Layout the dashboard 
dashboard = pn.column(pn.Row(price_spot, price_strike), pn.Row(maturity, risk_free_rate), pn.Row(volatility, option_type), update)

dashboard.servable()