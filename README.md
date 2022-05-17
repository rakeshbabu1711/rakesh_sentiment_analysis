# Sentiment Based Product Recommendation

### Problem Statement

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings.

In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.

- Data sourcing and sentiment analysis 
- Building a recommendation system 
- Improving the recommendations using the sentiment analysis model 
- Deploying the end-to-end project with a user interface

### Solution

* github link: https://github.com/rakeshbabu1711/rakesh_sentiment_analysis_capstone

* Heroku (Application is Live): https://rakeshsentiment-capstone-sbprs.herokuapp.com

### Built with

* Flask>=1.1.4
* nltk==3.7
* numpy==1.21.5
* pandas==1.4.2
* scikit-learn==1.1.0
* xgboost==1.6.1
* gunicorn>=20.1.0
* importlib-metadata==4.11.3
* itsdangerous==2.0.1
* jinja2<3.1.0
* MarkupSafe==2.0.1

### Solution Approach

* Dataset and Attribute description are available under "datasource" folder
* Dataset loading, exploratory data analysis, data cleaning and visualization, Text Pre-processing is performed on the dataset.
* TF-IDF Vectorizer is used to vectorize the textual data (review_title and review_text)
* During EDA, it is observed that the Dataset suffers from Class Imbalance Issue which is handled using SMOTE Oversampling technique before applying the model
* Machine Learning Classification Models (Logistic Regression, Naive Bayes, Tree Algorithms : (Decision Tree, Random Forest, XGBoost) are applied on the vectorized data and the target column (user_sentiment). the objective of this ML model is to classify the sentiment to positive(1) or negative(0). Best Model is selected based on the various ML classification metrics (Accuracy, Precision, Recall, F1 Score, AUC). XGBoost is selected to be a better model based on the evaluation metrics.
*  Collaborative Filtering Recommender system is created based on User-user and item-item approaches. RMSE evaluation metric is used for the evaluation.
*  Sentiment_analysis_rakesh_final.ipynb Jupyter notebook contains the code for Sentiment Classification and Recommender Systems
*  Top 20 products are filtered using the better recommender system, and for each of the products predicted the user_sentiment for all the reviews and filtered out the Top 5 products that have higher Postive User Sentiment (model.py)
*  Machine Learning models are saved in the pickle files(under the folder pickle\); Flask API (app.py) is used to interface and test the Machine Learning models. Bootstrap and Flask jinja templates (templates\index.html) are used for setting up the User interface.
*  End to End application is deployed in Heroku
