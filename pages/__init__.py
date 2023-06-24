# Regression Model Imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet

# Regression Metric Imports
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Classification Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier

# Classification Metric Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Clustering Imports
from sklearn.cluster import KMeans

# Other Necessary Imports
from sklearn.model_selection import train_test_split
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go


# Make Data
def make_data(data_frame,features,target):
    """Returns train test split with test size 20% and with random state as 404
    
    Keyword arguments:
    data_frame -- The Data Frame that needs to be splitted
    features -- The columns which are dependent
    target -- The column which is independent
    Return: X_train,X_test,y_train,y_test
    """
    
    X = data_frame[features]
    y = data_frame[target]
    return train_test_split(X,y,test_size=0.2,random_state=404)

# Apply Regression
reg_models = dict(
    linear = LinearRegression(),
    dt = DecisionTreeRegressor(),
    svr = SVR(),
    lasso = Lasso(),
    rf = RandomForestRegressor(),
    lgbm = LGBMRegressor(),
    xgbr = XGBRegressor(),
    net = ElasticNet(),
    br = BayesianRidge()
)
def apply_regression(data_frame,features,target):
    """Applies various ML Regression algorithms on the data frame
    
    Keyword arguments:
    data_frame -- The data frame that is used for ML
    reg_models -- The dictionary of various regression models
    Return: regression_data
    """
    # Split data into train and test
    X_train,X_test,y_train,y_test = make_data(data_frame=data_frame,features=features,target=target)
    regression_data = dict()
    models = dict()
    # Iterate over the models and fit them
    for model_name,model in reg_models.items():
        if model_name == 'xgbr':
            model.fit(X_train,y_train,verbose=0)            
        else:
            model.fit(X_train,y_train)
        y_preds = model.predict(X_test)
        r2,mae,mse = r2_score(y_test,y_preds),mean_absolute_error(y_test,y_preds),mean_squared_error(y_test,y_preds)
        regression_data[type(model).__name__] = dict(r2=r2,mae=mae,mse=mse)
        models[model_name] = model
    return regression_data,models

# Apply Classification
clf_models = dict(
    logistic = LogisticRegression(),
    svc = SVC(),
    gnb = GaussianNB(),
    mnb = MultinomialNB(),
    knn = KNeighborsClassifier(),
    dt = DecisionTreeClassifier(),
    rf = RandomForestClassifier(),
    lgbm = LGBMClassifier(),
    xgb = XGBClassifier()
)
def apply_classification(data_frame,features,target):
    # Split data into train and test
    X_train,X_test,y_train,y_test = make_data(data_frame=data_frame,features=features,target=target)
    clf_data = dict()
    models = dict()
    preds = dict()
    true = dict()
    # Iterate over the models and fit them
    for model_name,model in clf_models.items():
        if model_name == 'xgb':
            model.fit(X_train,y_train,verbose=0)
        else:
            model.fit(X_train,y_train)
        y_preds = model.predict(X_test)
        p,a,r,f1 = precision_score(y_test,y_preds,average="macro"),accuracy_score(y_test,y_preds),recall_score(y_test,y_preds,average="macro"),f1_score(y_test,y_preds,average="macro")
        clf_data[type(model).__name__] = dict(
            precision = p,
            accuracy = a,
            recall = r,
            f1 = f1
        )
        models[model_name] = model
        preds[model_name] = y_preds
        true[model_name] = y_test
    return clf_data, models, true, preds

def plot_reg(data_frame,features,target,model):
    _,X_test,_,y_test = make_data(data_frame=data_frame,features=features,target=target)
    y_pred = model.predict(X_test)
    fig = px.scatter(
        pd.DataFrame({'y_test':y_test,'y_pred':y_pred}),
        x = 'y_test',
        y = 'y_pred'
    )
    return fig

def plot_clf(data_frame,features,target,model):
    _,X_test,_,y_test = make_data(data_frame=data_frame,features=features,target=target)
    y_pred = model.predict(X_test)
    fig1,fig2 = px.histogram(
        pd.DataFrame({'y_test':y_test,'y_pred':y_pred}),
        x = 'y_test',
        color = 'y_test',
        text_auto=True
    ), px.histogram(
        pd.DataFrame({'y_test':y_test,'y_pred':y_pred}),
        x = 'y_pred',
        color = 'y_pred',
        text_auto = True
    )
    return fig1,fig2

def plot_elbow(data_frame):
    wcss = []
    for i in range(1,20):
        kmeans = KMeans(n_clusters=i,n_init="auto",init="k-means++",max_iter=300,random_state=42)
        kmeans.fit(data_frame)
        wcss.append(kmeans.inertia_)
    fig = px.line(x=range(1,20),y=wcss,markers=True,title="Elbow",labels={'x':'K','y':'Distortion Score'})
    return fig

def apply_clustering(data_frame,K):
    kmeans = KMeans(n_clusters=K,n_init="auto",random_state=42)
    kmeans.fit(data_frame)
    return kmeans

def plot_clustering(data_frame,model):
    df1 = data_frame
    df2 = pd.DataFrame(model.cluster_centers_,columns=data_frame.columns)
    df2['cluster'] = pd.Series(map(str,range(len(df2))))
    f1 = px.scatter(df1,df1.columns[0],df1.columns[1],color=model.labels_)
    f1.update_traces(marker_coloraxis=None)
    size = np.full_like(df2[df2.columns[0]],3)
    f2 = px.scatter(df2,df2.columns[0],df2.columns[1],text='cluster',color='cluster',symbol='cluster',size=size)
    layout = go.Layout(title="Clusters and their centers",xaxis=dict(title=df1.columns[0]),yaxis=dict(title=df1.columns[1]))
    f3 = go.Figure(data=f1.data+f2.data,layout=layout)
    f3.update_layout(height=500)
    return f3

# markdown
ml_md = r"""
# Machine Learning

### What is Machine Learning ?

> Computer systems can automatically learn from experience and get better over time thanks to a field of artificial intelligence called machine learning. In other words, it's a technique for teaching computers to recognize patterns, trends, and other information in data and to forecast the future or act on that knowledge. Object recognition in photos, comprehending spoken language, and making judgements based on complex data are just a few of the things that machine learning aims to make possible for machines to do that would otherwise require human intelligence.

### Types of Machine Learning

> There are three main types of machine learning: `Supervised`, `Unsupervised` and `Reinforcement` learning

### What is supervised learning?

> Supervised learning is a type of machine learning in which the algorithm is trained on a labelled dataset, which means that the input data has corresponding output data or labels. The goal is for the algorithm to learn the relationship between the input and output data so that it can accurately predict the output for new input data that it has not seen before. During training, the algorithm is essentially supervised by the labelled data, allowing it to make accurate predictions on unseen data in the future.

### What is unsupervised learning?

> The algorithm is trained on an unlabeled dataset in unsupervised learning, which is a type of machine learning where the input data does not have corresponding labels or output data. The algorithm must independently identify patterns or structure in the data without assistance or prior knowledge. Without being told what such patterns are or what the data represents, the computer learns to combine similar data points or spot underlying patterns. In essence, the algorithm is left to figure out what the underlying data structure is on its own.

### What is reinforcement learning ?

> An agent learns to make judgements through trial-and-error interactions with the environment in a type of machine learning known as reinforcement learning. By performing activities that result in favorable outcomes and avoiding those that result in undesirable ones, the agent aims to maximize a reward signal. The environment serves as the agent's teacher by giving feedback and sending signals of reinforcement for each action the agent takes. Following that, the agent modifies its behaviour to gradually increase the reward signal. In essence, the agent is rewarded or reinforced for wise choices, and with practise, develops superior judgement.

"""


sl_md = r"""
# Supervised Learning

A type of machine learning called supervised learning uses labelled examples supplied by a human expert to teach the algorithm how to map inputs to outputs. In other words, the algorithm employs the knowledge it has gained through training on a dataset that has already been labelled with the correct responses to make predictions on fresh, unlabeled data.

### Examples of supervised learning

* Spam detection
* Fraud detection
* House price prediction etc.

### Types of supervised learning

1. Regression
2. Classification
"""

lr_sl_md = r"""
The equation of a simple linear regression model can be expressed as:
$$
y = \beta_0 + \beta_1 + \epsilon
$$


Where:

- y is the dependent variable (also known as the response or outcome variable),
- x is the independent variable (also known as the predictor or explanatory variable),
- β0 is the intercept (the value of y when x is equal to 0),
- β1 is the slope (the change in y for a one-unit increase in x),
- ε is the error term (the random variation in y that cannot be explained by x).

The goal of linear regression is to estimate the values of β0 and β1 that best fit the data, by minimizing the sum of the squared residuals (the differences between the predicted and actual values of y). Once these values are estimated, the equation can be used to predict the value of y for any given value of x."""

r_sl_md = r"""
The equation of Ridge Regression is similar to that of simple linear regression, with the addition of a penalty term (λ) to the sum of squared residuals. The equation is as follows:
$$
y = \beta_0 + \beta_1.x_1 + .....+\beta_p.x_p + \epsilon
$$
where:

- y is the dependent variable
- x1, x2, ..., xp are the independent variables (also known as predictors or features)
- β0, β1, β2, ..., βp are the regression coefficients (the values to be estimated)
- ε is the error term

In Ridge Regression, the coefficients are estimated by minimizing the following objective function:
$$
RSS + \lambda . (\beta_1^2+\beta_2^2+.....+\beta_p^2)
$$
where RSS is the residual sum of squares (the sum of squared differences between the predicted and actual values of y), and λ is the regularization parameter that controls the strength of the penalty term. The regularization term penalizes large values of the coefficients, which helps to reduce overfitting and improve the generalization performance of the model.

The solution to this objective function can be found using various numerical optimization techniques, such as gradient descent or the normal equation. The resulting values of the coefficients can then be used to make predictions for new values of the independent variables."""

# endmarkdown
