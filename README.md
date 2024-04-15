# LLM-based Exchange Rate Forecasting
How can generative AI be used for exchange rate forecasting based on historical GDP, inflation, exchange rates, and other socioeconomic factors? In finance, an exchange rate is a measure between two currencies. At this rate, one currency is exchanged for another, i.e., exchange rates are determined in the foreign exchange market, which is open to buyers and sellers, and where currency trading is continuous. It is one of the major role players in economic decisions due to its importance and effect on all sorts of economic activities. Thus, exchange rate forecasting has a substantial role in economic decision-making. From this perspective, exchange rates are financial time series. 
The below graph shows that the rate fluctuates between 1.2 and 2.2, showing significant dependencies on the period. For example, Sterling can buy from 1.2$ to 2.2$: 

![](images/GBP.png)

It is quite understandable, why these periods are so much volatile. The first one is around 2008, which is the time of the sub-mortgage financial crisis and the second one is in 2016, 24 June, which is the day after the BREXIT referendum.

## Datasets
I have chosen the GBP/USD (British Sterling / US Dollar) rate to represent a financial time series for this project. I started with the primary data: GBP/USD exchange rate. Initially, I thought of collecting the daily exchange rate data from “https://www.investing.com/”. This source provides **Date, Price, Open, Height, Low, Volume and Change(%)**. Later on, I collected a new dataset from FED, including the other **22 exchange rate parities** (which I used for the multivariate analysis models) along with the **GBP/USD parity**.

    - Forex (daily -  AUD, EUR, NZD, GBP, BRL, CAD, CNY, DKK, HKD, INR, JPY, MYR, MXN, NOK, ZAR, SGD, KRW, LKR, SEK, CHF, TWD, THB, VEB.), from https://www.federalreserve.gov/, from 2000-01-03 to 2020-08-21.

My second data source was the Federal Reserve Bank of St. Louis. I have collected **interest rates** and **normalized GDP** data (both for the US and UK, data downloaded separately) from here. 
    
    - Normalized GDP (monthly), from https://fred.stlouisfed.org/ from 2000-01-01 to 2024-04-13.     
    - Libor Rates (daily), from https://fred.stlouisfed.org/, from 2001-01-02 to 2020-09-18.

Lastly, I acquired current account to GDP data from the OECD’s website and current account to GDP (quarterly), from https://stats.oecd.org/, from Q1-2000 to Q1-2020. Eventually, I had 29 different data (28 of them to be predictor variables) and 4920 observations for each of them. I saved my pre-processed data to a new `CSV` file: **"Merged.csv"**

  - Foreign Exchange Rates
  - GDP
  - Current Account to GDP
  - Libor Rates (Interest Rates for GBP and USD)

## Methodological approaches 
In terms of modelling forecast methods, it is essential to understand the features of exchange rates to develop decent models. First and foremost, exchange rates are sequenced data. The transactions are executed sequentially w.r.t with a timestamp, yielding a time series. Besides their sequenced nature, their other notable feature is that they are nonlinear and nonstationary, meaning they are nondirectional and ever-changing without presenting any regularity. Literature suggests some statistical (or econometric) modelling methods such as (S)ARIMA, ETS (for univariate series), or VAR (for multivariate series) to be employed. ARIMA methodology is used as a baseline model, while MLP and CNN architectures will use DL methods. Besides, 4 different **DL** architectures were used to compare with MLP Univariate, CNN Univariate, MLP Multivariate, and CNN Multivariate. Key findings suggest that one of the **DL** architectures was good enough to outperform the ARIMA process. 

### Base Model (ARIMA)
I intended to create a base model using the econometric Autoregressive Integrated Moving Average (ARIMA). We read the data and extracted the GBP series for univariate analysis.

![](images/ARIMA.png)

For the test set, I separated the last 50 observations of the data. The prediction vs actual comparison graph looks good. We evaluated the model w.r.t RMSE. I observed a Test RMSE of 0.00643: the lower the RMSE, the better the prediction is. The residual distribution is very close to a normal distribution, confirming that we have done our job in this ARIMA process correctly. 

### MLP Univariate
We model Multi-Layer Perceptron for UNIVARIATE analysis in a univariate analysis setting, e.g., we use yesterday’s data to predict today and today’s data to predict tomorrow and so on. This means, we used the same data shifted … a lagged value of itself. We can use data that is 2 or 3 or more days earlier. This is a level that we decide for the lag value. I predicted 50 days’ data, thus I decided to use a lag value for 50 days, though I could have chosen my lag value differently, too: 

![](images/MLP_U.png)

I observed a Test RMSE of 0.01295, which is OKish to say that there is a good match, yet the evaluation metric is not as good as the ARIMA process.

### CNN Univariate
The second DL architecture is CNN in a univariate setting. When evaluating the model on the test set, following similar data preprocessing, I observed an RMSE is 0.03028. The score is 2,5 times worse than the MLP_U model.

![](images/CNN_U.png)

### MLP Multivariate
For the multivariate analysis using MLP architecture, we lowered the lagged values to 5 from 50 compared to the previous models. Yet there are 28 more variables in the equation. 22 of them being other exchange rates and the rest are main macroeconomic indicators for both countries US and the UK. The test score for MLP_M seems a tad bit better than the CNN_U model: 0.0273: 

![](images/MLP_M.png)

### CNN Multivariate
Time-series being a sequenced data provides a good opportunity to test CNN architectures abilities -- especially for multivariate analysis. This model seems more successful than its univariate counterpart, yielding an RMSE metric of 0.02076: 

![](images/CNN_M.png)

## Conclusion
When we compare the results we can see that the literature findings are correct. The base model for the prediction has the best metric by far. While the result of the base model is 0.00643, the closest result to that one is 0.01295 from the Univariate MLP architecture. Then comes the CNN multivariate prediction process with 0.02076 and the MLP multivariate process with 0.02739. The CNN univariate process has the worst score with 0.03028. The ARIMA process is a linear regressor, thus the results of the process would be the same even if we repeat it, yet that cannot be told for the deep learning processes. In all of these deep learning processes, random factors are used, therefore the results of each process change each time it runs. Whereas it can get better, it is similarly probable that they can get worse. And actually, I had my presentation video for this project before finishing the README.md and the results that I have in that video are quite different from these. At this point, I believe an alternative could be feeding the different groups of variables separately, to this CNN architecture, by using functional API. It is a useful tool, especially in such cases where different variables are presented and affect the response in different ways.
