## **Financial Sentiment Analysis**
We download news articles using newsapi Python client related to the Eurozone or EUR currency from the NewsAPI portal. For this, I wrote a method that fetches news articles for the specified date. Following search terms are used in the query: 

  * "European Commission" 
  * "EU" 
  * "European Central Bank" 
  * "ECB" 
  * "Eurozone" 
  * "EUR currency".

Then, I used a transformer language model (LLM) from HuggingFace's transformers library to generate sentiment scores for the downloaded articles. The sentiments for each article's title and content are then calculated and added to the DataFrame. 

Finally, I'll extend it using the gradio library such that I can take input a headline of article via interface and predict the sentiment using the LLM.
