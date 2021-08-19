---
layout: post
title: First Blog Post
---

After cloing Barry Clark's [Jekyll Now Repo](https://github.com/barryclark/jekyll-now), I have created my first blog post with it. This was written as an exercise for a Data Science Bootcamp. We are wrapping up, with less than one week to go, and it's an appropriate time to learn how to create and use our ever important portfolios. I look forwards to the writing I'll be posting here or my personal site!

Here's an example of my work

---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Modeling

We will use multiple models, optimize hyperparameters, and feature engineer to guage and boost the accuracy of our Natural Language Processing.

## Table of Contents

* [0. Data Dictionary](#chapter0)
    * [0.1 Basic EDA](#section0_1)
* [1. Linear Classifier](#chapter1)
    * [1.1 Fit with `lbfgs`](#section1_1)
        * [1.1.1 Choosing Scaler](#section1_1_1)
    * [1.2 Most Predictive Words](#section1_2)
    * [1.3 PCA](#section1_3)
    * [1.4 Advantages and Disadvantages of Dimensionality Reduction](#section1_4)
* [2. K-Nearest Neighbor Classifier](#chapter2)
    * [2.1 Fit](#section2_1)
        * [2.1.1 Choosing Scaler](#section2_1_1)
    * [2.2 Reduce Observations](#section2_2)
    * [2.3 Advantages and Disadvantages of Observation Reduction](#section2_3)
    * [2.4 Optimal K in KNN Algorithm](#section2_4)
        * [2.4.1 Choosing Scalers Better](#section2_4_1)
    * [2.5 Issues with Splitting after Vectorization](#section2_5)
* [3. Decision Tree Classifier](#chapter3)
    * [3.1 Fit](#section3_1)
        * [3.1.1 Choosing Scaler](#section3_1_1)
    * [3.2 Optimal Maximum Depth of Decision Tree](#section3_2)
    * [3.3 Advantages and Weaknesses of Decision Trees](#section3_3)
* [4+5. Cross Validation](#chapter4and5)
    * [4 Purpose of Validation Set](#section4)
    * [5 Re-running Decision Tree](#section5)
        * [5.1 5 Cross Fold Validation and Optimizing Hyperparameters](#section5_1)
        * [5.2 Confusion Matrix](#section5_2)
* [6. Feature Engineering](#chapter6)
    * [6.1 Feature Explanation](#section6_1)
    * [6.2 Rerunning Previous Decision Tree Model](#section6_2)
* [7. Conclusion](#chapter7)

## Data Dictionary<a class="anchor" id="chapter0"></a>

- __Additional_Number_of_Scoring__: An extra metric for scoring
- __Average_Score__: Average of all review scores received
- __days_since_review__: Number of days since review was posted
- __lat__: Geographic Latitude of hotel
- __lng__: Geographic Longitude of hotel
- __Review_Total_Negative_Word_Counts__: Word count of negative review
- __Review_Total_Positive_Word_Counts__: Word count of positive review
- __Total_Number_of_Reviews__: Number of reviews hotel received
- __Total_Number_of_Reviews_Reviewer_Has_Given__: How many reviews were posted by reviewer in total


- __weekday_of_review__: The weekday the review was posted
- __month_of_review__: The month the review was posted 
- __year_of_review__: The year the review was posted
- __rating__: Binary column denoting good ratings as 1 and bad ones as 0.
    * Note: Good rating of 1 represents scores of 9 and 10, bad ratings of 0 represent scores of 1 through 8

### Basic EDA<a class="anchor" id="section0_1"></a>

We have been given clean data to work with; it has been neatly split into train and test sets so we can focus on modeling. However, as it is best practice, we still perform fundamental EDA.

train_data = pd.read_csv('data/train_dataframe.csv')

test_data = pd.read_csv('data/test_dataframe.csv')

train_data.head()

test_data.head()

print(f'Train data has {train_data.shape[1]} columns and {train_data.shape[0]} rows')
print(f'Train - Missing values: {train_data.isna().sum().sum()}')
print(f'Train - Duplicated rows: {train_data.duplicated().sum()}')
print(f'Test data has {test_data.shape[1]} columns and {test_data.shape[0]} rows')
print(f'Test - Missing values: {test_data.isna().sum().sum()}')
print(f'Test - Duplicated rows: {test_data.duplicated().sum()}')
print(f'This represents a test-train split of {round(test_data.shape[0]/(test_data.shape[0]+train_data.shape[0]), 2) * 100.00}%')

There are no immediate errors to be fixed, we can continue exploring the nature of the data

# Cursory glance at columns
train_data.columns

# First 15 columns checked to find numeric columns before vectorized columns
for i in range(0,15):
    print(train_data.columns[i])

train_data['rating'].value_counts()

values, counts = np.unique(train_data['rating'], return_counts=True)
normalized_counts = counts/counts.sum()

plt.figure()
plt.bar(values, normalized_counts * 100)
plt.xlabel('Review rating')
plt.ylabel('% of reviews')
sns.despine()
plt.title("Rating distribution")
plt.show()

Lastly, we take a look at `rating` distribution. Dividing the rating between "good" as 1 and "bad" as 0, helps us create a somewhat balanced distribution. It seems that our target is still skewed towards "good" ratings despite only scores of 9 and 10 being deemed "good".

## 1. Linear Classifier<a class="anchor" id="chapter1"></a>

The first model we take a look at is Logistic Regression. We will not be doing any hyperparameter optimization, and instead use the model to study dimensionality reduction as well as the model coefficients and visualize it's most predictive words.

from sklearn.linear_model import LogisticRegression

X_train = train_data.drop('rating', axis=1)

y_train = train_data['rating']

X_test = test_data.drop('rating', axis=1)

y_test = test_data['rating']

### 1.1 Fit with `lbfgs`<a class="anchor" id="section1_1"></a>

#### 1.1.1 Choosing Scaler<a class="anchor" id="section1_1_1"></a>

To start, we will be scaling our data as is best practice. We will begin by comparing StandardScaler, MinMaxScaler, and RobustScaler.

from sklearn.preprocessing import StandardScaler

# Scale data using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_ss = scaler.transform(X_train)
X_test_ss = scaler.transform(X_test)

from sklearn.preprocessing import MinMaxScaler

# Scale data using MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(X_train)
X_train_mm = scaler.transform(X_train)
X_test_mm = scaler.transform(X_test)

from sklearn.preprocessing import RobustScaler

# Scale data using RobustScaler
scaler = RobustScaler()
scaler = scaler.fit(X_train)
X_train_rob = scaler.transform(X_train)
X_test_rob = scaler.transform(X_test)

scales = [[X_train, X_test, "Unscaled"],[X_train_ss, X_test_ss, "StandardScaler"],[X_train_mm, X_test_mm, "MinMaxScaler"], [X_train_rob, X_test_rob, "RobustScaler"]]

for scaler in scales:
    print(f'{scaler[2]}:')
    # Instantiate
    # Set max_iter to 1000, default is 100; but this reduces
    # ConvergenceWarning
    hotel_logit = LogisticRegression(solver='lbfgs', max_iter=1000)

    # Fit
    hotel_logit.fit(scaler[0], y_train)

    # Score
    print(f'Train Score: {hotel_logit.score(scaler[0], y_train)} \n')

We choose to scale our data with StandardScaler. We pick our scaler based only off of train score to avoid leakage from test data. Personally, I do know the test scores and I know MinMaxScaler has better fit to them; but as a __principled__ stance, we continue with StandardScaler (We will work with validation data further on, which will give us much better insight on the best scaler to use)

# Instantiate
hotel_logit = LogisticRegression(solver='lbfgs', max_iter=1000)

# Fit
hotel_logit.fit(X_train_ss, y_train)

# Score
print(f'Train Score: {hotel_logit.score(X_train_ss, y_train)}')

# Accuracy on the test set
print(f'Test Score: {hotel_logit.score(X_test_ss, y_test)}')

Our accuracy score on the test set is 73%, showing how powerful base logistic regression can be.

### 1.2 Most Predictive Words<a class="anchor" id="section1_2"></a>

Next, we take a look at what our model calculates as the most predictive words. We may be able to determine real world insights on review sentiment with these.

positive_words=[]
negative_words=[]
words=[]
for i in train_data.columns:
    if str(i)[0:2] == 'p_':
        positive_words.append(str(i))
        words.append(str(i))
    elif str(i)[0:2] == 'n_':
        negative_words.append(str(i))
        words.append(str(i))
    else:
        pass

Broad look into the positive and negative words from our working dataframe.

positive_words

negative_words

# Cursory glance at coefficient scale
pd.DataFrame(hotel_logit.coef_)

To visualize our most predictive words, we need to slice our coefficient array to just the coefficients for the words. Then we can sort from highest (more correlated with ratings of 1) to lowest (more correlated with ratings of 0)

len(words)

len(np.array(hotel_logit.coef_)[0][417:])

coef_logit = np.array(hotel_logit.coef_)[0][417:]

def plot_coefs(coef_logit, words):
    coef_df = pd.DataFrame({"coefficient": coef_logit, "token": words})
    print(type(coef_df["token"]))
    coef_df = coef_df.sort_values("coefficient", ascending=False)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # smallest coefficient -> tokens indicating negative sentiment 
    coef_df.tail(20).set_index("token").plot(kind="bar", rot=45, ax=axs[0], color="red")
    axs[0].set_title("Negative indicators")
 
    
    # largest coefficient -> tokens indicating positive sentiment 
    coef_df.head(20).set_index("token").plot(kind="bar", rot=45, ax=axs[1], color="blue")
    axs[1].set_title("Positive indicators")
    
    sns.despine()
    plt.tight_layout()
    plt.show()
    
plot_coefs(coef_logit, words)

We have the option to add duplicate words such as "n_staff"/"p_staff" to stopwords; as they show up often in both positive and negative reviews. They both show an absolute value correlation of 0.4, meaning negative reviews and positive reviews are equally likely to comment on staff.

There are other interesting insights, for example `n_inn` being vectorized from negative reviews is the strongest coefficient as a positive indicator. Many of the strongest Positive Indicator coefficients are words from negative reviews, possibly indicating that reviews with those words still gave a 9-10 in rating.  

### 1.3 PCA<a class="anchor" id="section1_3"></a>

Next, we take a look at how dimensionality reduction may affect our model. We will be using PCA and rerunning our Logistic Regression with 90% of the variance.

from sklearn.decomposition import PCA

# Instantiate
my_pca = PCA(n_components=0.9)

# Fit and Transform
my_pca.fit(X_train_ss)

# transform data 
X_train_PCA = my_pca.transform(X_train_ss)
X_test_PCA = my_pca.transform(X_test_ss)

X_train_PCA.shape

Our feature space has been reduced to 1891, from a previous size of 2326 features. This maintains 90% of variance and reduces the amount of data our model processes by almost 20%!

# New logistic Regression with Reduced Dimensions
my_logreg_PCA = LogisticRegression()

# Fitting to PCA data
my_logreg_PCA.fit(X_train_PCA,y_train)

# Scoring on PCA train and test sets
print(f'Train Score: {my_logreg_PCA.score(X_train_PCA, y_train)}')
print(f'Test Score: {my_logreg_PCA.score(X_test_PCA, y_test)}')

We were only expecting a boost in run-time speed and comparable performance, but we also see higher test score and better fit. I believe this is coincidence as PCA in theory has little bearing on model accuracy. However, external research has shown PCA boosting model accuracy in practice.

### 1.4 Advantages and Disadvantages of Dimensionality Reduction<a class="anchor" id="section1_4"></a>

Advantages:
- Speeds up computation time
- Similar model performance

Disadvantages:
- Reduces interpretability of coefficients (since one PCA feature is a calculation of all other features)

## 2. K-Nearest Neighbors<a class="anchor" id="chapter2"></a>

The second model we will be taking a look at is K-Nearest Neighbors. Here we will be optimizing hyperparameters and performing observation reduction as well.

from sklearn.neighbors import KNeighborsClassifier

### 2.1 Fit<a class="anchor" id="section2_1"></a>

#### 2.1.1 Choosing Scaler

 As KNN is a distance-based model, scaling data is absolutely essential. We will be taking the same approach as we did in Chapter 1 (using the same array `scaler` as it remained unchanged)

for scaler in scales:
    print(f'{scaler[2]}:')
    # Instantiate
    hotel_knn = KNeighborsClassifier()

    # Fit
    hotel_knn.fit(scaler[0], y_train)

    # Score
    print(f'Train Score: {hotel_knn.score(scaler[0], y_train)} \n')

This time we choose to go with RobustScaler. Again, we are not sure of overfit, but will continue with the scaler that gives us the highest training score: RobustScaler.

# Instantiate
hotel_knn = KNeighborsClassifier()

# Fit
hotel_knn.fit(X_train_rob, y_train)

# Score
print(f'train score: {hotel_knn.score(X_train_rob, y_train)}')
print(f'test score: {hotel_knn.score(X_test_rob, y_test)}')

With an accuracy score on the test set if 72%, our KNN model did perform slightly worse than our Logistic Regression model. However, the test data does show significantly better fit with the train data here.

### 2.2 Reduce Observations <a class="anchor" id="section2_2"></a>

As KNN is computationally expensive, we will try reducing observations to boost computation speed.

# Use pandas.sample to get 80% of the data
train_data_reduced = train_data.sample(frac=0.8)

test_data_reduced = test_data.sample(frac=0.8)

train_data_reduced.shape

test_data_reduced.shape

# New feature and target arrays to feed into model
X_train_reduced = train_data_reduced.drop('rating', axis=1)

y_train_reduced = train_data_reduced['rating']

X_test_reduced = test_data_reduced.drop('rating', axis=1)

y_test_reduced = test_data_reduced['rating']

# Instantiate
hotel_knn = KNeighborsClassifier()

# Fit
hotel_knn.fit(X_train_reduced, y_train_reduced)

# Score
print(f'train score: {hotel_knn.score(X_train_reduced, y_train_reduced)}')

Our unscaled reduced dataset loses very little train score accuracy. We will now scale our reduced data with RobustScaler and then compare run times.

# Scale with RobustScaler
scaler = RobustScaler()
scaler = scaler.fit(X_train_reduced)
X_train_reduced_rob = scaler.transform(X_train_reduced)

%%timeit
# Instantiate
hotel_knn = KNeighborsClassifier()

# Fit
hotel_knn.fit(X_train, y_train)

# Score
hotel_knn.score(X_train, y_train)

hotel_knn.score(X_train, y_train)

%%timeit
# Instantiate
hotel_knn = KNeighborsClassifier()

# Fit
hotel_knn.fit(X_train_reduced, y_train_reduced)

# Score
hotel_knn.score(X_train_reduced, y_train_reduced)

hotel_knn.score(X_train_reduced, y_train_reduced)

hotel_knn.score(X_test_reduced, y_test_reduced)

Just an 80% reduction in sample size boosts our overall performance significantly. On my computer, the mean run time went from well over 10 seconds with the entire dataset to around 4-7 seconds with the reduced data.

We may see an improved train accuracy score as well, but this is most likely due to chance, with the sample distribution of the data being more convenient with the KNN algorithm.

However, the nail in the coffin is the test score reduction. It seems by reducing our number of observations, we have sacrificed many model insights.

### 2.3 Advantages and Disadvantages of Observation Reduction<a class="anchor" id="section2_3"></a>

Advantage
- Faster Computation Speeds

Disadvantage
- Potentially lose valuable model insights

### 2.4 Optimal K in KNN Algorithm<a class="anchor" id="section2_4"></a>

With our reduced dataset showing faster computational speeds, we can put these times to the test as we optimize for `n_neighbors`. We start by creating a validation set to optimize without any data leakage.

X_train_reduced, X_val, y_train_reduced, y_val = train_test_split(X_train_reduced, y_train_reduced, test_size = 0.2, stratify=y_train_reduced)

X_train_reduced.shape

y_train_reduced.shape

#### 2.4.1 Choosing Scalers Better<a class="anchor" id="section2_4_1"></a>

# Scale data using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_reduced)
X_train_ss = scaler.transform(X_train_reduced)
X_val_ss = scaler.transform(X_val)

# Scale data using MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(X_train_reduced)
X_train_mm = scaler.transform(X_train_reduced)
X_val_mm = scaler.transform(X_val)


# Scale data using RobustScaler
scaler = RobustScaler()
scaler = scaler.fit(X_train_reduced)
X_train_rob = scaler.transform(X_train_reduced)
X_val_rob = scaler.transform(X_val)

scales = [[X_train_reduced, X_val, "Unscaled"],[X_train_ss, X_val_ss, "StandardScaler"],[X_train_mm, X_val_mm, "MinMaxScaler"], [X_train_rob, X_val_rob, "RobustScaler"]]
                                                                                   
for scaler in scales:
    print(f'{scaler[2]}:')
    # Instantiate
    hotel_knn = KNeighborsClassifier()

    # Fit
    hotel_knn.fit(scaler[0], y_train_reduced)

    # Score
    print(f'Train Score: {hotel_knn.score(scaler[0], y_train_reduced)}')
    print(f'Validation Score: {hotel_knn.score(scaler[1], y_val)} \n')

We will continue optimizing for `n_neighbors` with RobustScaler. I admit, I did not expect the fitting for StandardScaler and MinMaxScaler to be so poor. I do get the feeling I misunderstood the process of scaling based off of validation data, but to keep in line with best practices; we will continue with RobustScaler.

train_scores = []
validation_scores = []
k_range = range(1,50)

for k in k_range:
    # Instantiate
    hotel_knn = KNeighborsClassifier(n_neighbors=k)

    # Fit
    hotel_knn.fit(X_train_rob, y_train_reduced)

    # Score
    train_scores.append(hotel_knn.score(X_train_rob, y_train_reduced))
    validation_scores.append(hotel_knn.score(X_val_rob,y_val))

plt.figure()
plt.plot(k_range, train_scores,label="Train Score",marker='.')
plt.plot(k_range, validation_scores,label="Validation Scores",marker='.')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show();

After careful examination of our plot, I believe an `n_neighbors` of 18 will provide the best fit for our data. This is because our validation scores are highest there, and the model is not significantly overfit.

# Instantiate
hotel_knn = KNeighborsClassifier(n_neighbors=18)

# Fit
hotel_knn.fit(X_train_rob, y_train_reduced)

# Score
print(f'Test Score: {hotel_knn.score(X_test_reduced, y_test_reduced)}')

As our training data set has been further reduced (X_train_rob is 80% of X_train_reduced), we managed show a 2% increase test score just with hyperparameter optimization.

We know that model Test accuracy suffers with reduced observation, so the fact that training our model with even smaller data has improved Test accuracy shows the performance boost an `n_neighbors` of 18 can provide.

### 2.5 Issues with Splitting after Vectorization<a class="anchor" id="section2_5"></a>

Credit: https://towardsdatascience.com/3-things-you-need-to-know-before-you-train-test-split-869dfabb7e50

The purpose of `train_test_split()` is to get two datasets that are completely independent one another to maintain the integrity of our model. We take many careful measures to ensure testing data does not leak into our training data.

However, when splitting __after__ vectorization, we inherently leak train data into our test data. As a vectorizer is fit on the whole dataset, whatever data will be assigned to our test set later, will naturally carry information from that initial vectorization. This is why it is best practice to split our data first and then vectorize.

## 3. Decision Tree<a class="anchor" id="chapter3"></a>

The last model that we will be taking a look at is Decision Tree. We will be optimizing for `max_depth` and using validation data to do so.

from sklearn.tree import DecisionTreeClassifier

### 3.1 Fit<a class="anchor" id="section3_1"></a>

#### 3.1.1 Choosing Scaler<a class="anchor" id="section3_1_1"></a>

Since we are working with Decision Trees, we do not need any scaler. Decision Trees by nature of the algorithm are not sensitive to scale, and will make the same decisions whether the data is scaled or not.

# Instantiate
hotel_dt = DecisionTreeClassifier()

# Fit
hotel_dt.fit(X_train_reduced, y_train_reduced)

# Score
print(f'Train Score: {hotel_dt.score(X_train_reduced, y_train_reduced)}')
print(f'Validation Score: {hotel_dt.score(X_val, y_val)}')

As we are working with Decision Trees, it is natural for our Train Score to be 100% (as the training data is memorized). A validation score of 68% shows massive overfitting, but we can sacrifice some train data memorization by reducing `max_depth` and produce better overall model fit.

### 3.2 Optimal Maximum Depth of Decision Tree<a class="anchor" id="section3_2"></a>

We use a similar optimization process for `max_depth` in `DecisionTreeClassifier` as we did with `n_neighbors` in `KNeighborsClassifier`

train_scores = []
validation_scores = []
c_range = range(1,10)

for c in c_range:
    # Instantiate
    hotel_dt = DecisionTreeClassifier(max_depth=c)

    # Fit
    hotel_dt.fit(X_train_reduced, y_train_reduced)

    # Score
    train_scores.append(hotel_dt.score(X_train_reduced, y_train_reduced))
    validation_scores.append(hotel_dt.score(X_val,y_val))

plt.figure()
plt.plot(c_range, train_scores,label="Train Score",marker='.')
plt.plot(c_range, validation_scores,label="Validation Scores",marker='.')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show();

We choose a `max_depth` of 5 as that is where our validation socre peaks while our model maintains a relatively small fit.

# Instantiate
hotel_dt = DecisionTreeClassifier(max_depth=5)

# Fit
hotel_dt.fit(X_train_reduced, y_train_reduced)

# Score
print(f'Train Score: {hotel_dt.score(X_train_reduced, y_train_reduced)}')
print(f'Test Score: {hotel_dt.score(X_test_reduced, y_test_reduced)}')

With a test score of 75%, our model has performed excellently. The fit of our model is also fantastic, as it is less than 1% off from our train score.

### 3.3 Advantages and Weaknesses of Decision Trees <a class="anchor" id="section3_3"></a>

Advantages
- As the KNN algorithm compares every feature of every data point, it will not be able to weigh those features, whereas Decision Trees can
- Scaling with KNN can be tricky and requires careful consideration. Decision Trees are insensitive to scaling.

Disadvantages
- 
- The process of using Decision Trees for regression means that our model can never make predictions larger than the maximum value in our training set.

## 4&5. Cross Validation<a class="anchor" id="chapter4and5"></a>

from sklearn.model_selection import cross_val_score

### 4. Purpose of Validation Set <a class="anchor" id="chapter4"></a>

The purpose of the validation set is to prevent data leakage. For example, one important use case is with hyperparameter optimization. Picking the best hyperparameter involves comparing fit of the model. However, we can not use the test data judge model fit, as basing our model on test data biases our model to that test data, and is a clear leak.

Using a validation set, we can freely guage and adjust model fit, knowing that the hyperparameters we found are based solely off of the training data.

### 5. Re-running Decision Tree <a class="anchor" id="chapter5"></a>

With that insight on Decision Trees, we can use a powerful tool, Cross Fold Validation, to further improve our hyperparameter optimization process.

#### 5.1 5 Cross Fold Validation and Optimizing Hyperparameters<a class="anchor" id="section5_1"></a>

# Instantiate
hotel_dt_cv = DecisionTreeClassifier()

# Fit with 5 cross fold validation
scores = cross_val_score(hotel_dt_cv, X_train_reduced, y_train_reduced, cv = 5)
print(scores)

cross_validation_scores = []
c_range = range(1,10)

for c in c_range:
    hotel_dt = DecisionTreeClassifier(max_depth=c)
    cv_score = np.mean(cross_val_score(hotel_dt, X_train_reduced, y_train_reduced, cv = 5))
    cross_validation_scores.append(cv_score)

plt.figure()
plt.plot(c_range, cross_validation_scores,label="Cross Validation Score",marker='.')
plt.legend()
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Cross Validation Score')
plt.grid()
plt.show();

which_max = np.array(cross_validation_scores).argmax()

print("The best model has k = ",c_range[which_max])

As you can see, the best mean validation score comes from a `max_depth` of 8.  We fit our model accordingly.

# Instantiate
hotel_dt = DecisionTreeClassifier(max_depth=5)

# Fit
hotel_dt.fit(X_train_reduced, y_train_reduced)

# Score
print(f'Train Score: {hotel_dt.score(X_train_reduced, y_train_reduced)}')
print(f'Test Score: {hotel_dt.score(X_test_reduced, y_test_reduced)}')



#### 5.2 Confusion Matrix <a class="anchor" id="section5_2"></a>

from sklearn.metrics import plot_confusion_matrix, confusion_matrix

# Get class predictions
y_pred = hotel_dt.predict(X_test_reduced)

# Generate confusion matrix
cf_matrix = confusion_matrix(y_test_reduced, y_pred)

# label rows and columns
cf_df = pd.DataFrame(
    cf_matrix, 
    columns=["Predicted Bad Rating", "Predicted Good Rating"],
    index=["True Bad Rating", "True Good Rating"]
)

display(cf_df)

# Precision
print(f'Model Precision: {1545/(1545+429)}')

# Recall
print(f'Model Recall: {1545/(1545+417)}')

# FPR
print(f'False Positive Rate: {429/(1023+429)}')

# TPR
print(f'True Positive Rate: {1 - 429/(1023+429)}')

Our model's recall, precision, and accuracy all seem to be in line. Taking those extra measure to ensure we have balanced classes has proved useful.

plot_confusion_matrix(hotel_dt, X_test_reduced, y_test_reduced);

## 6. Feature Engineering<a class="anchor" id="chapter6"></a>

A simple and effective feature we can add is a positive/negative word ratio. While the `Review_Total_Negative_Word_Counts` and `Review_Total_Positive_Word_Counts` columns may add some insights to our model independently, I believe the information in those features will be much more effective as a ratio.

train_data_2 = train_data.copy()

# Replace 0's with 1's
# We will be dividing by Review_Total_Positive_Word_Counts
# We cannot divide by 0
train_data_2['Review_Total_Positive_Word_Counts'] = train_data_2['Review_Total_Positive_Word_Counts'].replace(0,1)

train_data_2['Review_Word_Count_Ratio'] = train_data_2['Review_Total_Negative_Word_Counts']/train_data_2['Review_Total_Positive_Word_Counts']

train_data_2.head()

train_data_2 = train_data_2.drop(['Review_Total_Negative_Word_Counts','Review_Total_Positive_Word_Counts'], axis=1)

# Check
train_data.head()

test_data_2 = test_data.copy()

test_data_2['Review_Total_Positive_Word_Counts'] = test_data_2['Review_Total_Positive_Word_Counts'].replace(0,1)

test_data_2['Review_Word_Count_Ratio'] = test_data_2['Review_Total_Negative_Word_Counts']/test_data_2['Review_Total_Positive_Word_Counts']

test_data_2['Review_Word_Count_Ratio'].max()

test_data_2 = test_data_2.drop(['Review_Total_Negative_Word_Counts','Review_Total_Positive_Word_Counts'], axis=1)

### 6.1 Feature Explanation<a class="anchor" id="section6_1"></a>

It is not too useful to score the number of words in total independently, does not seem indicative of sentiment by itself.

For example, let's say we have one reviewer that wrote 10 words for their negative review and another that wrote 50 words for their negative review. Our model may be inclined to weigh more words as more likely to be a bad rating, and less words more likely to be a good rating. However, consider then, if the first reviewer wrote only 5 words for their positive review and the second reviewer wrote 100 words for their positive review. The weight the model makes indepently on the negative reviews word count gets counterracted by the behavior of the positive review word count.

This is why using a ratio would provide more insights than judging the two independently.

### 6.2 Rerunning Previous Decision Tree Model<a class="anchor" id="section6_2"></a>

Using our new feature, we re rung the previous Decision Tree Model

train_data_2_reduced = train_data_2.sample(frac=0.8)

test_data_2_reduced = test_data_2.sample(frac=0.8)

train_data_2_reduced.shape

test_data_2_reduced.shape

X_train_2_reduced = train_data_2_reduced.drop('rating', axis=1)

y_train_2_reduced = train_data_2_reduced['rating']

X_test_2_reduced = test_data_2_reduced.drop('rating', axis=1)

y_test_2_reduced = test_data_2_reduced['rating']

X_test_2_reduced['Review_Word_Count_Ratio'].max()

hotel_dt_2 = DecisionTreeClassifier()

hotel_dt_2.fit(X_test_2_reduced, y_test_2_reduced)

# Instantiate
hotel_dt_cv = DecisionTreeClassifier()

# Fit with 5 cross fold validation
scores = cross_val_score(hotel_dt_cv, X_train_2_reduced, y_train_2_reduced, cv = 5)
print(scores)

cross_validation_scores = []
c_range = range(1,10)

for c in c_range:
    hotel_dt = DecisionTreeClassifier(max_depth=c)
    cv_score = np.mean(cross_val_score(hotel_dt, X_train_2_reduced, y_train_2_reduced, cv = 5))
    cross_validation_scores.append(cv_score)

plt.figure()
plt.plot(c_range, cross_validation_scores,label="Cross Validation Score",marker='.')
plt.legend()
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Cross Validation Score')
plt.grid()
plt.show();

which_max = np.array(cross_validation_scores).argmax()

print("The best model has k = ",c_range[which_max])

# Instantiate
hotel_dt = DecisionTreeClassifier(max_depth=6)

# Fit
hotel_dt.fit(X_train_2_reduced, y_train_2_reduced)

# Score
print(f'Train Score: {hotel_dt.score(X_train_2_reduced, y_train_2_reduced)}')
print(f'Test Score: {hotel_dt.score(X_test_2_reduced, y_test_2_reduced)}')

We see marginal improvement. While not drastic, for one feature to show countable improvement does mean that we are on the right track. I believe the model is making more predictive insights with `Review_Word_Count_Ratio`

## 7. Conclusion<a class="anchor" id="chapter7"></a>

Through this exercise, we have run three separate models to analyze sentiment and predict rating.

We have practice dimensionality reduction as well as observation reduction.

We have also practiced using validation sets to optimize hyperparameters.

Lastly, we did feature engineering and showed improvement in our model accuracy.