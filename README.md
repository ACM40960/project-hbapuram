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

Please note that in order to use the code, you must have a reddit developer account and the API keys to retrieve data. The keys referenced in the code have been masked. The documentation for PRAW can be found [here](https://praw.readthedocs.io/en/stable/)

The Financial Data obtained for this project was sourced from [Yahoo Finance](https://finance.yahoo.com/) and the files thus obtained are uploaded in the `FINANCIAL DATA` folder in [RAW DATA](https://github.com/ACM40960/project-hbapuram/tree/main/RAW%20DATA)

## Reddit Data Pre-Processing and Sentiment Analysis

> The code for this section can be refered to from the `data_preprocess_text.ipynb` file found [here]() in the Python Files directory. Specific parts/functions are oulined in the summary below:

The texual data prior to any analysis, required extensive cleaning and reformatting and the process can be outlined in the following steps as executed by the `preprocess_text()` function:

- Check if 'text' is a non-null string
- Remove punctuation
- Convert to lowercase
- Tokenize text (i.e., separate the words in a sentence into individual tokens to analyse)
- Remove stopwords
- Filter out short words
- Join tokens back into a string

Some initial cleaning was applied to restructure the `headlines` and `comments` files in order to extract ony relevant variables for analysis prior to merging all the textual files for a company into one common dataframe.

Post Analysis, the processed text looks something similar to the output obtained below:

![Text-preprocess-output](https://github.com/ACM40960/project-hbapuram/blob/main/Images/text-preprocess-ex-1.png)

### Sentiment Analysis

In order to perform sentiment analysis on the text, we first filter the data for the required time range and create functions to extract the **Subjectivity Score** (from `Text Blob`) **Compound Score** (from `VADER`) which will form the basis of the Sentiment Score used in prediction. 

Post this analysis, we can also obtain a wordcloud to explore the most frequently used terms in the conversations about a particular company. Here's an example for Apple:

![Apple Wordcloud](https://github.com/ACM40960/project-hbapuram/blob/main/Images/apple_wordcloud.png)

The sentiment scores thus obtained using the `getPolarity(text)` and `getSIA(text)` functions are then aggregated for each day by taking a weighted average of the score weighted by the `score` variable from Reddit which is indicative of the popularity of each comment. The end result for each day would thus be an average of the scores obtained for all text published on that day with more importance given to texts with greater popularity/iteraction (implied by higher score) and vice versa.

Before we proceed to merge the results here with the financial data, we `Forward Fill` the days with missing observations for each company. The final processed text file will have the following structure:

- 123 rows corresponding to each day of the target date range
- 5 columns corresponding to `date`, `score`, `w_subj` (weighted subjectivity score), `w_polar` (weighted polarity score) and `w_comp` (weighted compound score)

 Please note that the polarity score and compound score are comprable - both are metrics which convey the extent of 'positivity' or 'negativity' detected in a sentence. The polarity metric is obtained from the `Text Blob` library and the compound score from `VADER`. For the purpose of our study, we chose the compund scoe as the metric to convey sentiment since VADER has been found to perform better in analysing social media content. 

## Financial Data Pre-processing

> The code for this section can be refered to from the `data_preprocess_fin.ipynb` file found [here]() in the Python Files directory. Specific parts/functions are oulined in the summary below:

The financial data does not require any implicit cleaning since the data in very well structured. The only pre-processing performed here was the application of Forward fill to get continuous data without any missing observations in the date range. The data was also filtered for the target date range. The final processed text file will have the following structure:

- 123 rows corresponding to each day of the target date range
- 7 columns corresponing to `date`, `open`, `high`, `low`, `close`, `adj_close` (which is the response variable in our prediction model) and `volume`

## Merging the financial data and the Sentiment Scores by Date

> The preprocessed files were then merged together by date using the concat() function in python. The code for the same can be found the the `merge_text_fin.ipynb` file and the combined data files can be found the in the [DATA](https://github.com/ACM40960/project-hbapuram/tree/main/DATA) directory

## Prediction Model using RNN LSTM

### Predicting Adjusted Close Price without Sentiment Scores

### Predicting Adjusted Close Price with Sentiment Scores

### Model Evaluation using RMSE, MAE, MAPE and Validation Loss

## Results and Conclusion

- The results obtained upon the implementation of the prediction algorithm for Adjusted Closing Price, we notice that the inclusion of the Sentiment Parameters makes a positive impact on improving performance
  
- The model fit didn't require a lot of hyperparameter tuning, with only minor adjustments made to the **number of input units**, **number of epochs**, and **dropout rate**.
  
- The sentiment analysis scores weighted by Reddit Score (Given as Number of Upvotes - Number of Downvotes) as the weight component in calculating the aggregate sentiment score for each day accounted for the level of interaction on a particular comment/headline thereby evaluating the average sentiment for each day appropriately in accordance with the "influence" of the particular text.
  
- Sentiment features, i.e., Subjectivity Score and Compound score prove to be particularly impactful in predicting future values where there have been huge spikes/dips (as observed in behaviour of Microsoft, Nvidia, and Tesla) since the improvement in the evaluation measures of Root Mean Squared Error, Mean Absolute Error and Mean Absolute Percentage Error has been greater in these cases than the rest.
  
- As an extended study, evaluating the data and analysing the unpredictable highs and lows might facilitate for the creation of a simulation to generate worst-case scenarios for companies to use while conducting stress tests or developing Business Continuity Plans.

## Poster:

[Poster Link](https://github.com/ACM40960/project-hbapuram/blob/main/PROJECT%20POSTER%20HRISHITA%20.pdf)

## Literature Review:

[Literature Review Link](https://github.com/ACM40960/project-hbapuram/blob/main/Literature_Review_Hrishita_project.pdf)

## Acknowledgements

I want to extend my heartfelt gratitude to Dr. Sarp Akcay for the invaluable guidance and unwavering support provided during the duration of the module at University College Dublin. My gratitude also extends to University College Dublin for furnishing the necessary resources that played a pivotal role in realizing this project. Finally, I wish to express my deep thanks to all online resources including referred publications that made the learning required for this project easy and accessible for everyone.












 







