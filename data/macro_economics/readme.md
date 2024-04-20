# Data sources and metadata 
We describe each type of data along with some domain-specific metadata description. 

## Macroeconomic data 
Macroeconomic data refers to the broad economic statistics that represent the overall health and efficiency of an economy. This data encompasses a wide range of indicators 
such as national accounts, inflation rates, unemployment rates, balance of payments, fiscal indicators, trade statistics, and commodity prices. These indicators are crucial 
for understanding the economic performance of countries or regions and are often used by governments, financial institutions, and researchers to make informed decisions.

When it comes to time series forecasting tasks like currency exchange rates, macroeconomic data plays a pivotal role. Exchange rates are influenced by numerous factors, including 
economic policies, market sentiment, political stability, and macroeconomic indicators. Forecasters use this data to understand past trends and to make predictions about future movements in exchange rates.
For instance, if a country's inflation rate is rising, it might lead to a depreciation of its currency because the purchasing power of the currency is eroding. Similarly, if a country has a strong balance 
of payments surplus, it might indicate that more foreign currency is entering the country than leaving, which could strengthen the currency. 

Moreover, some studies have shown that long-maturity forward exchange rates can be used as a proxy for the fundamental equilibrium exchange rate, which is the rate that would exist if 
markets were in perfect balance. By comparing the long-maturity forward rate to the actual exchange rate, forecasters can estimate the gap and make predictions about future movements in the exchange rate.
In summary, macroeconomic data is essential for understanding the underlying forces that drive currency exchange rates and can be used in various statistical models to forecast future rates. 
These forecasts are valuable for investors, multinational corporations, and policymakers who need to make decisions based on expected changes in exchange rates.

## Sources and description 
Those data were downloaded from ....

  - GRCP20YY Index: "Consumer prices (CPI) are a measure of prices paid by consumers for a market basket of consumer goods and services. The yearly (or monthly) growth rates represent the inflation rate."
  - GRIORTMM Index: "This concept tracks the volume of new orders received during the reference period. Orders are typically based on a legal agreement between two parties in which the producer will deliver goods or services to the purchaser at a future date."
  - EURR002W Index: "A target interest rate set by the central bank in its efforts to influence short-term interest rates as part of its monetary policy strategy. This indicator shows the new target interest rate on the date the new rate was announced. 
    The main refinancing rate is the rate for the Eurosystem's regular open market operations (in the form of a reverse transaction) to provide the banking system with the amount of liquidity that the former deems to be appropriate. Main refinancing operations are conducted through weekly standard tenders (in which banks can bid for liquidity) and normally have a maturity of one week."    "This concept tracks the volume of new orders received during the reference period. Orders are typically based on a legal agreement between two parties in which the producer will deliver goods or services to the purchaser at a future date."
    
### How these data can be used 
Forecasters may use statistical models like ARIMA (Autoregressive Integrated Moving Average) and GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models to analyze and predict currency exchange rates. 
These models can incorporate macroeconomic data to account for the economic factors that influence exchange rates. For example, an ARIMA model might use historical exchange rate data along with inflation differentials 
between two countries to forecast future exchange rates. 


