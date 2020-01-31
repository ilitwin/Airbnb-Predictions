---
title: 'Classification'
author: "Izabela Litwin"
output:
  pdf_document:
    toc: yes
    toc_depth: 3
  html_document:
    df_print: paged
    toc: yes
    toc_depth: '3'
---

```{r setup, echo=TRUE, include=FALSE}
knitr::opts_chunk$set(echo = TRUE) 
knitr::opts_chunk$set(warning = FALSE) 
knitr::opts_chunk$set(message = FALSE) 
```

```{r setup2, echo=TRUE, include=FALSE}
library(kableExtra)
library(tidyverse)
library(pander)
library(Hmisc)
library(xtable)
library(rpart)
```

Reviews

```{r}
review = read.csv('review_cleaned.csv')
review = review[,-(which(names(review) == "X"))]
review = review[,-(which(names(review) == "amenities"))]
review = review[,-(which(names(review) == "latitude"))]
review = review[,-(which(names(review) == "longitude"))]
review$review_scores_rating = as.factor(review$review_scores_rating)
```


```{r}
TestSet = read.csv("review_cleaned_test.csv")
TestSet = TestSet[,-(which(names(TestSet) == "X"))]
TestSet = TestSet[,-(which(names(TestSet) == "amenities"))]
TestSet = TestSet[,-(which(names(TestSet) == "latitude"))]
TestSet = TestSet[,-(which(names(TestSet) == "longitude"))]
TestSet$review_scores_rating = as.factor(TestSet$review_scores_rating)
```


```{r}
library(caret)
library(doParallel) 
detectCores()
registerDoParallel(cores=4)

# automated parameter tuning of C5.0 decision tree 
set.seed(300)

## Customizing the tuning process ----
# use trainControl() to alter resampling strategy
ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "oneSE")

# use expand.grid() to create grid of tuning parameters
grid <- expand.grid(.model = "tree",
                    .trials = c(1, 5, 10, 15, 20, 25, 30, 35),
                    .winnow = "FALSE")

# look at the result of expand.grid()
grid


m <- train(review_scores_rating ~ ., data = review, method = "C5.0",
           metric = "Kappa",
           trControl = ctrl,
           tuneGrid = grid)

m

# auto-tune a boosted C5.0 decision tree
grid_c50 <- expand.grid(.model = "tree",
                        .trials = c(10, 20, 30, 40),
                        .winnow = "FALSE")

m_c50 <- train(review_scores_rating ~ ., data = review, method = "C5.0",
                metric = "Kappa", trControl = ctrl,
               tuneGrid = grid_c50)
m_c50

## KNN

knn <- train(review_scores_rating ~ ., data=review, method = "knn",
                    preProcess = c("center", "scale"), 
                    tuneGrid=expand.grid(k=1:50),
                    trControl = trainControl(method = "cv", number=5))

knn

## Bagged trees 

ctrl <- trainControl(method = "cv", number = 5)

baggedtree = train(review_scores_rating ~ ., data = review, method = "treebag",
      trControl = ctrl)

baggedtree

## Random Forests

library(caret)
ctrl <- trainControl(method = "repeatedcv",
                     number = 3, repeats = 3)

# auto-tune a random forest
grid_rf <- expand.grid(.mtry = c(4, 8, 16, 32))

m_rf <- train(review_scores_rating ~ ., data = review, method = "rf",
              metric = "Kappa", trControl = ctrl,
              tuneGrid = grid_rf)
m_rf


## Adaboost 

## Using AdaBoost.M1
library(adabag)

# create a Adaboost.M1 model
set.seed(300)
m_adaboost <- boosting(review_scores_rating ~ ., data = review)
p_adaboost <- predict(m_adaboost, review)
head(p_adaboost$class)
p_adaboost$confusion

# create and evaluate an Adaboost.M1 model using 5-fold-CV

adaboost_cv <- boosting.cv(review_scores_rating ~ ., data = review, v = 5)
adaboost_cv$confusion

1-adaboost_cv$error

# calculate kappa
library(vcd)
Kappa(adaboost_cv$confusion)


# Naive Bayes

naive.bayes.model <- train(x = review[,-23], 
                           y = as.factor(review$review_scores_rating), 
                           data = review, 
                           trControl = trainControl(method = "cv", number = 5),
                           method = "nb")

naive.bayes.model

# Important variables according to naive bayes model
naive.bayes.imp <- varImp(naive.bayes.model)
print(naive.bayes.imp)
plot(naive.bayes.imp, top = 20, main = "Naive Bayes Important features")

# prediction on test data
# testData <- prepare_testData(naive.bayes.model$trainingData[, -ncol(naive.bayes.model$trainingData)],test.dtm)

naive.bayes.pred <- predict(naive.bayes.model, TestSet)

# error metrics
naive.bayes.metrics <- confusionMatrix(naive.bayes.pred, TestSet$review_scores_rating)
print(naive.bayes.metrics)


### Support Vector Machines

# Training svm model
svm.model <- train(review_scores_rating ~., 
                    data = review, 
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    trControl = trainControl(method = "cv", number = 5),
                    tuneLength = 8)

svm.model

# Important variables according to SVM model
svm.imp <- varImp(svm.model)
print(svm.imp)
plot(svm.imp, top = 20, main = "SVM Important features")

# prediction on test data 
svm.pred <- predict(svm.model, TestSet)

# error metrics
svm.metrics <- confusionMatrix(svm.pred, TestSet$review_scores_rating)
print(svm.metrics)

## Logistic Regression

glm.model <- train(review_scores_rating ~., 
                   data = review, 
                   method = "glm", family = binomial(link = "logit"),
                   trControl = trainControl(method = "cv", number = 5))

glm.model

# Important variables according to glm model
glm.imp <- varImp(glm.model)
print(glm.imp)
plot(glm.imp, top = 20, main = "Logistic Regression Important features")

# prediction on test data
glm.pred <- predict(glm.model, TestSet, type = "raw")

# Error metrics
glm.metrics <- confusionMatrix(glm.pred, TestSet$review_scores_rating)
print(glm.metrics)

## Decision tree 

tree.model <- train(review_scores_rating ~., 
                           data = review, 
                           trControl = trainControl(method = "cv", number = 5),
                           method = "ctree")

tree.model

# Important variables according to ctree model
tree.imp <- varImp(tree.model)
print(tree.imp)
plot(tree.imp, top = 20, main = "Decision Tree Important features")

# prediction on test data
tree.pred <- predict(tree.model, TestSet, type = "raw")

# Error metrics
tree.metrics <- confusionMatrix(tree.pred, TestSet$review_scores_rating)
print(tree.metrics)

## Neural Networks 

#will take a few minutes
nnet.model <- train(review_scores_rating ~.,  
                    data = review, 
                    trControl = trainControl(method = "cv", number = 5),
                    method='pcaNNet')
plot(nnet.model)

nnet.model

# Important variables according to nn model
nnet.imp <- varImp(nnet.model)
print(nnet.imp)
plot(nnet.imp, top = 20, main = "Neural Network Important features")

# prediction on test data
nnet.pred <- predict(nnet.model, TestSet, type = "raw")

# Error metrics
nnet.metrics <- confusionMatrix(nnet.pred, TestSet$review_scores_rating)
print(nnet.metrics)


## Ensemble training - Majority Vote
# Takes as input the prediction results from:
#   1. Naive Bayes
#   2. SVM
#   3. Logistic Regression
#   4. Decision Tree
#   5. Neural Networks
# Predicts the value voted by majority of classifiers mentioned above

pred_ensemble <- function(nb.pred, svm.pred, glm.pred, tree.pred, nnet.pred) {
  ensemble.pred <- rep(0, length(nb.pred))
  for(i in 1:length(ensemble.pred)) {
    # case 1: If more than two classifiers vote as positive
   if (sum(as.numeric(levels(nb.pred[i])[nb.pred[i]]), 
           as.numeric(levels(svm.pred[i])[svm.pred[i]]),
           as.numeric(levels(glm.pred[i])[glm.pred[i]]),
           as.numeric(levels(tree.pred[i])[tree.pred[i]]),
           as.numeric(levels(nnet.pred[i])[nnet.pred[i]]))  >= 3) {
     ensemble.pred[i] <- 1
   }
    # Else - keep the value zero (more than two classifiers vote is negative)
  }
  return(ensemble.pred)
}

# prediction using ensemble method
ensemble.pred <- pred_ensemble(naive.bayes.pred, svm.pred, glm.pred, 
                               tree.pred, nnet.pred)

# Error metrics
ensemble.metrics <- confusionMatrix(as.factor(ensemble.pred), 
                                    TestSet$review_scores_rating)
print(ensemble.metrics)

```

Testing

```{r}
# C5.0 decision tree
preds = predict(m, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# C5.0 decision tree (auto tuning)
preds = predict(m_c50, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# KNN
preds = predict(knn, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# Bagged Trees
preds = predict(baggedtree, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# Random Forests 
preds = predict(m_rf, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# Adaboost
preds = predict(m_adaboost, TestSet)$class
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# Naive Bayes
preds = predict(naive.bayes.model, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# SVM
preds = predict(svm.model, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)
 
# Logistic Regression
preds = predict(glm.model, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# Decision Tree
preds = predict(tree.model, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# Neural Networks 
preds = predict(nnet.model, TestSet)
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)

# Ensemble 
preds = ensemble.pred
confusionMatrix(as.factor(preds), TestSet$review_scores_rating)
```
