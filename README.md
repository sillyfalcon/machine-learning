# machine-learning-walmart-store-sales-prediction
Checkout the Jupyter Notebook here: https://jovian.ai/anushree-k/final-walmart-simple-rf-gbm

<b>Business Problem Statement</b>:
- Predicting department-wide weekly sales for each Walmart store
- Predict which departments are affected and the extent of the impact due to holiday markdowns based on limited history

<b>Evaluation criteria and loss functions</b>:
  <b>WMAE</b> - weighted mean absolute error
  
<b>Brief Description:</b>
  In this notebook we will explore Supervised Machine Learning methods. Regression models such as linear regression, decision tree and ensemble models such as RandomForest, XGBoost, LightGBM will trained to predict weekly sales using Scikit Learn, LightGBM and XGBoost. We will use Pandas, Numpy, Matplotlib, Seaborn and Plotly to perform exploratory data analysis and gather insights for machine learning. We will do the following

- Install and Import libraries
- Explore the dataset and merge different files as required
- Translate the business problem to a machine learning problem
- EDA - exploratory data analysis
- Feature Engineering
- Data preparation - Train Val Split, Encoding, Imputing and Scaling
- Select input features
- Define evaluation metrics
- Define baseline model
- Select best model without hyperparameter tuning
- Hyperparameter tuning for select models
- Make predictions
- Save the best model
- Summarise insights and learning

<b>About the Dataset</b>
This Dataset is from the Walmart Recruiting - Store Sales Forecasting competition on Kaggle.
- It has 4 files train.csv, test.csv, store.csv and features.csv all totaling 5 GB in size
- This dataset has 536,634 rows ( train = 421,570, test = 115,064) and 14 columns  
Kaggle_url: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

