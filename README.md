# Data Science Contest (Loan Targeting Optimization Using Deep Learning and Time Series Modelling)

One of the major issues almost all commercial banks face is how to decide whom to offer a loan. In this data science contest, one of the largest banks in South Korea, Hana Bank, proposed a problem of identifying small business owners from its existing customer base and selectively advertising to them loans through a pop-up ad or survey ad using information such as gender, age, region, login and login duration data. To solve the problem, we applied traditional statistical techniques such as SARIMA models and modern machine learning algorithms such as Neural Networks and CNNs. At the end of the contest, we were able to place among the top 15 teams out of a total of 40 participating teams from UNIST, KAIST, and POSTECH.

To understand the proposed challenge and read about the dataset description, please refer the "Contest Guide" file.

We first performed EDA and data-preprocssing to understand the differences between small business owners and non-business owners.

For Taks 1, to reduce number of parameters that need to be trained, we transformed daily time series data into weekly format experimented with different ML models for classification such as Random Forest, Neural Networks, CNNs and TabTransformer. After that we performed cross validation and chose the best model.

For Task 2, we algorithmically chose the optimal threshold for classifying business owners and we decided to showed a pop-up to all business owners chosen based on that threshold.

For Task 3, we kept daily login data and used two seasonal ARIMA models with frequencies of 7 (weekly) and 30 (monthly), and then took their maximum prediction for the future 5 days (we say that he or she is predicted to login if the maximum prediction is more than or equal to 1 for one of the following 5 days). We decided to show a survey ad if a person is classified as a small business onwer and he or she is predicted to login between August 27-31.

To see the details of our solution, please take a look at the code files.

Here is the overall leaderboard (our team name is **Gold Diggers**):

![](https://github.com/eldor-fozilov/data-science-contest/blob/main/leaderboard.png)

