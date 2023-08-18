# Stock Price Prediction using Sentiment Analysis of Reddit Text

## Introduction: Reddit and Alternative Data in Stock Performance Analysis

In recent times, Reddit has surfaced as a favored platform for engaging in conversations and deliberations spanning a broad array of subjects, which encompass financial matters and investments. With many users sharing their viewpoints and understandings concerning diverse stocks and financial markets, Reddit has transformed into a valuable information pool for gauging sentiments within the financial domain. Additionally, with the emergence of Alternative Data as a growing source to evaluate Market Performance, studying the impact of market sentiment could add an invaluable measure while studying possible vulnerabilities of a stock.

This project delves into the application of sentiment analysis for assessing the fluctuations of widely-held stocks, employing Reddit headlines and comments as a wellspring of market sentiment. This project aims to scrutinize pre-existing studies that leverage sentiment analysis derived from Reddit to approximate market volatility, and consequently, future gains.

## Authors

[@Hrishita-Bapuram](https://www.github.com/hbapuram) - 22204557

[@Soujanya-Hassan-Prabhakar](https://www.github.com/soujanya-hassan-prabhakar) - 22202225

## Installation

Following is the exhaustive list of all the libraries used throughout the course of the project:

```python
!pip install numpy
!pip install pandas
!pip install plotly
!pip install matplotlib
!pip install statsmodels
!pip install seaborn
!pip install sklearn
!pip install scipy
!pip install datasets >> NULL
!pip install transformers >> NULL
!pip install swifter >> NULL
!pip install -U sentence-transformers
!pip install -U scikit-learn >> NULL
!pip install vaderSentiment
!pip install wordcloud
import nltk
nltk.download('all')
```
## Project Flow

The structure and flow of the project can be summarized in the following flow chart:

![Project Flow](https://github.com/ACM40960/project-hbapuram/blob/main/Images/project-flow.png)

## Data Extraction

The Reddit Data obtained for this project was extracted from their `PRAW` package. While the extracted files are available in [RAW DATA](https://github.com/ACM40960/project-hbapuram/tree/main/RAW%20DATA) for each company under their respective company folders, the code used for extraction has been uploaded in the [Python Files](https://github.com/ACM40960/project-hbapuram/tree/main/Python%20Files) directory as `reddit_data_scrapping.ipynb` linked [here](https://github.com/ACM40960/project-hbapuram/blob/main/Python%20Files/reddit_data_scrapping.ipynb)

Please note that in order to use the code, you must have a Reddit developer account and the API keys to retrieve data. The keys referenced in the code have been masked. The documentation for PRAW can be found [here](https://praw.readthedocs.io/en/stable/)

The Financial Data obtained for this project was sourced from [Yahoo Finance](https://finance.yahoo.com/) and the files thus obtained are uploaded in the `FINANCIAL DATA` folder in [RAW DATA](https://github.com/ACM40960/project-hbapuram/tree/main/RAW%20DATA)

## Reddit Data Pre-Processing and Sentiment Analysis

> The code for this section can be referred to from the `data_preprocess_text.ipynb` file found [here]() in the Python Files directory. Specific parts/functions are outlined in the summary below:

The textual data prior to any analysis, required extensive cleaning and reformatting and the process can be outlined in the following steps as executed by the `preprocess_text()` function:

- Check if 'text' is a non-null string
- Remove punctuation
- Convert to lowercase
- Tokenize text (i.e., separate the words in a sentence into individual tokens to analyze)
- Remove stopwords
- Filter out short words
- Join tokens back into a string

Some initial cleaning was applied to restructure the `headlines` and `comments` files in order to extract all relevant variables for analysis prior to merging all the textual files for a company into one common dataframe.

Post Analysis, the processed text looks something similar to the output obtained below:

![Text-preprocess-output](https://github.com/ACM40960/project-hbapuram/blob/main/Images/text-preprocess-ex-1.png)

### Sentiment Analysis

In order to perform sentiment analysis on the text, we first filter the data for the required time range and create functions to extract the **Subjectivity Score** (from `Text Blob`) **Compound Score** (from `VADER`) which will form the basis of the Sentiment Score used in prediction. 

Post this analysis, we can also obtain a wordcloud to explore the most frequently used terms in the conversations about a particular company. Here's an example for Apple:

![Apple Wordcloud](https://github.com/ACM40960/project-hbapuram/blob/main/Images/apple_wordcloud.png)

The sentiment scores thus obtained using the `getPolarity(text)` and `getSIA(text)` functions are then aggregated for each day by taking a weighted average of the score weighted by the `score` variable from Reddit which is indicative of the popularity of each comment. The end result for each day would thus be an average of the scores obtained for all texts published on that day with more importance given to texts with greater popularity/interaction (implied by higher score) and vice versa.

Before we proceed to merge the results here with the financial data, we `Forward Fill` the days with missing observations for each company. The final processed text file will have the following structure:

- 123 rows corresponding to each day of the target date range
- 5 columns corresponding to `date`, `score`, `w_subj` (weighted subjectivity score), `w_polar` (weighted polarity score), and `w_comp` (weighted compound score)

 Please note that the polarity score and compound score are comparable - both are metrics that convey the extent of 'positivity' or 'negativity' detected in a sentence. The polarity metric is obtained from the `Text Blob` library and the compound score from `VADER`. For the purpose of our study, we chose the compound score as the metric to convey sentiment since VADER has been found to perform better in analyzing social media content. 

## Financial Data Pre-processing

> The code for this section can be referred to from the `data_preprocess_fin.ipynb` file found [here]() in the Python Files directory. Specific parts/functions are outlined in the summary below:

The financial data does not require any implicit cleaning since the data is very well structured. The only pre-processing performed here was the application of Forward fill to get continuous data without any missing observations in the date range. The data was also filtered for the target date range. The final processed text file will have the following structure:

- 123 rows corresponding to each day of the target date range
- 7 columns corresponding to `date`, `open`, `high`, `low`, `close`, `adj_close` (which is the response variable in our prediction model), and `volume`

## Merging the financial data and the Sentiment Scores by Date

> The preprocessed files were then merged together by date using the concat() function in Python. The code for the same can be found the the `merge_fin_text.ipynb` file and the combined data files can be found in the [DATA](https://github.com/ACM40960/project-hbapuram/tree/main/DATA) directory

## Prediction Model using RNN LSTM

We have employed the `LSTM (Long Short-Term Memory)` model to predict the adjusted close price of a stock based on its historical data and sentiment scores. The LSTM model is initialized with one or more LSTM layers, followed by a dropout layer to prevent overfitting, and a dense output layer to produce the final prediction. The model is compiled with an optimizer (Adam) and a loss function (mean squared error), and trained on the training dataset using early stopping, learning rate reduction, and model checkpoint techniques to prevent overfitting and improve performance.

Here's an example of setting up the LSTM model for predicting the adjusted close price for AAPL stocks:

```python
# setting the model architecture
# Initializing the Neural Network based on LSTM
model = Sequential()
# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=False, input_shape=(len(cols), 1)))
# Adding Dropout
model.add(Dropout(0.25))
# Output layer
model.add(Dense(units=1, activation='linear'))

# Compiling the Neural Network
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

# fitting the model using the training dataset
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=15, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')

# printing the model summary
model.summary()
```

![Model Summary](https://github.com/ACM40960/project-hbapuram/blob/main/Images/Model_Summary.png)

![Hyperparameters](https://github.com/ACM40960/project-hbapuram/blob/main/Images/HyperParameters.png) 

For each company, we've tailored the LSTM model parameters to achieve optimal results. These hyperparameters are tuned to achieve optimal performance for each company's stock price prediction.

### Predicting Adjusted Close Price without Sentiment Scores

> The code for this section can be referred to from the `Prediction_Model_without_Sentiment.ipynb` file found [here](https://github.com/ACM40960/project-hbapuram/blob/main/Python%20Files/Prediction_Model_without_Sentiment.ipynb) in the Python Files directory. Specific parts/functions are outlined in the summary below:

To predict the Adjusted Close Price without incorporating sentiment scores, we will focus solely on the historical financial data. Specifically, we will consider columns containing stock prices including `open`, `high`, `low`, and `volume`. Our goal is to build an LSTM-based model to forecast the stock's Adjusted Close Price, leveraging the temporal patterns in the historical price data.

### Predicting Adjusted Close Price with Sentiment Scores

> The code for this section can be referred to from the `Prediction_Model_with_Sentiment.ipynb` file found [here](https://github.com/ACM40960/project-hbapuram/blob/main/Python%20Files/Prediction_Model_with_Sentiment.ipynb) in the Python Files directory. Specific parts/functions are outlined in the summary below:

In this section, we will enhance our stock price prediction model by incorporating sentiment scores derived from the analysis of Reddit text data. By combining financial data with sentiment analysis, we aim to investigate whether sentiment signals from social media can improve the accuracy of our stock price predictions.
To include sentiment scores, we're utilizing the combined dataset of the pre-processed sentiment data with historical financial data which will serve as the input for our prediction model. Our approach involves predicting the Adjusted Close Price using variables such as `open`, `high`, `low`, and `volume`, in addition to sentiment-related metrics like `w_subj` (weighted subjectivity score) and `w_comp` (weighted compound score).

In this visual representation, we observe the significant fluctuations within the data when sentiment scores are absent, as opposed to the distinct influence that sentiments exert in enhancing the precision of stock price predictions.

![Prediction_Graphs](https://github.com/ACM40960/project-hbapuram/blob/main/Images/Prediction_Graphs.png)

### Model Evaluation using RMSE, MAE, MAPE, and Validation Loss

Below are the metrics depicting the evaluation of the models with and without sentiment scores for various companies:

![Model_Evaluation](https://github.com/ACM40960/project-hbapuram/blob/main/Images/Model_Evaluation.png)

These metrics and loss values provide insight into how the models perform with and without the incorporation of sentiment scores, for each respective company. The comparison aids in understanding the influence of sentiment analysis on the prediction accuracy of stock prices.

## Results and Conclusion

- The results obtained upon the implementation of the prediction algorithm for Adjusted Closing Price, we notice that the inclusion of the Sentiment Parameters makes a positive impact on improving performance
  
- The model fit didn't require a lot of hyperparameter tuning, with only minor adjustments made to the **number of input units**, **number of epochs**, and **dropout rate**.
  
- The sentiment analysis scores weighted by Reddit Score (Given as Number of Upvotes - Number of Downvotes) as the weight component in calculating the aggregate sentiment score for each day accounted for the level of interaction on a particular comment/headline thereby evaluating the average sentiment for each day appropriately in accordance with the "influence" of the particular text.
  
- Sentiment features, i.e., Subjectivity Score and Compound score prove to be particularly impactful in predicting future values where there have been huge spikes/dips (as observed in the behavior of Microsoft, Nvidia, and Tesla) since the improvement in the evaluation measures of Root Mean Squared Error, Mean Absolute Error and Mean Absolute Percentage Error has been greater in these cases than the rest.
  
- As an extended study, evaluating the data and analyzing the unpredictable highs and lows might facilitate the creation of a simulation to generate worst-case scenarios for companies to use while conducting stress tests or developing Business Continuity Plans.

## Poster:

[Poster Link](https://github.com/ACM40960/project-hbapuram/blob/main/PROJECT%20POSTER%20HRISHITA%20.pdf)

## Literature Review:

[Literature Review Link](https://github.com/ACM40960/project-hbapuram/blob/main/Literature_Review_Hrishita_project.pdf)

## Acknowledgements

I want to extend my heartfelt gratitude to Dr. Sarp Akcay for the invaluable guidance and unwavering support provided during the duration of the module at University College Dublin. My gratitude also extends to University College Dublin for furnishing the necessary resources that played a pivotal role in realizing this project. Finally, I wish to express my deep thanks to all online resources including several publications that made the learning required for this project easy and accessible for everyone.












 







