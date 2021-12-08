---
  title: "R Notebook"
output: html_notebook
---
  
  
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(caret)
library(ggplot2)
library(dplyr)


# Read Data


data = read.csv("data4.csv", na.strings = c("","NA"))
head(data)


#missing values


sum(is.na(data)) #na value count
data$Property1[is.na(data$Property1)] = mean(data$Property1, na.rm = TRUE) #filling mean values for property1 in first 10 rows where na
head(data)


#deleting NA data for Property 2 Predictions


data_2 = na.omit(data) #omit na. Now 330 rows




data_2$Property2 = ifelse(data_2$Property2=="yes",1,0) #1 if Property2 = yes
str(data_2) #internal structure. check for property2
#data_2$Property2 = as.factor((data_2$Property2))



#proper fit values for Property 2
#splitting the data of 330 rows into 80-20

set.seed("500")
split <- createDataPartition(data_2$Property2,p=0.8,list=FALSE) #splitting to 80% and 20%
train_Property2 <- data_2[split,]
test_Property2 <- data_2[-split,]


# Regression to get values for Property2


lrmodel_property2 = lm(data=train_Property2,Property2~.)
lrmodel_property2
summary(lrmodel_property2)

# predict_Linear_Property2_initial = predict(lrmodel_property2,newdata = test_Property2)
# predict_Linear_Property2_initial
# predict_Linear_Property2_initial = ifelse(predict_Linear_Property2_initial>=0.5,1,0)
# predict_Linear_Property2_initial

#confusionMatrix(factor(predict_Linear_Property2_initial,levels = 0:1),factor(test_Property2$Property2,levels = 0:1))


#finding predicted values for Property 2


predict_Property2 = predict(lrmodel_property2,newdata = test_Property2, type = 'response')
predict_Property2
predict_Property2 = ifelse(predict_Property2>=0.5,1,0)
predict_Property2

confusionMatrix(factor(predict_Property2,levels = 0:1),factor(test_Property2$Property2,levels = 0:1))


#Replacing NA VALUES TO 1 since Mode is 1


data$Property2 = ifelse(data$Property2=="yes",1,0)
str(data_2)
data$Property2[is.na(data$Property2)] = 1
head(data)


#YESSSS DATA FOUND


#PROPERTY 2

#Feature selection 


library(randomForest)
best_feature = randomForest(data$Property1~.,data = data)
importance(best_feature) #9,11,12


#Modelling

#Split data (4 variable dataset) into train and test


data_final_Property2 <- data[,c("Property2","Feature9","Feature11","Feature12")]

set.seed("1001")
#splitting the 4 variables
split_main_Property2 <- createDataPartition(data_final_Property2$Property2,p=0.8,list=FALSE)
train_main_Property2 <- data_final_Property2[split_main_Property2,]
test_main_Property2 <- data_final_Property2[-split_main_Property2,]


#Logistic Regression


log_model_pro2 = glm(data = train_main_Property2,formula = Property2~., family = binomial)
summary(log_model_pro2)





#prediction for logistic regression


predict_Logistic_Property2 = predict(log_model_pro2,newdata = test_main_Property2, type = 'response')
predict_Logistic_Property2
predict_Logistic_Property2 = ifelse(predict_Logistic_Property2>=0.5,1,0)
predict_Logistic_Property2


#Confusion matrix - accuracy


confusionMatrix(factor(predict_Logistic_Property2,levels = 0:1),factor(test_main_Property2$Property2,levels = 0:1))


#Boosting

#
str(train_main_Property2)
boosting_model_pro2 = train(data = train_main_Property2,Property2~., method = 'gbm',verbose=FALSE)
summary(boosting_model_pro2)

# predicting and confusion matrix

#
predict_Boosting_Property2 = predict(boosting_model_pro2,newdata = test_main_Property2)
predict_Boosting_Property2
predict_Boosting_Property2 = ifelse(predict_Boosting_Property2>=0.5,1,0)
predict_Boosting_Property2

confusionMatrix(factor(predict_Boosting_Property2,levels = 0:1),factor(test_main_Property2$Property2,levels = 0:1))


# Bagging


bagging_model_pro2 = train(data = train_main_Property2,Property2~., method = 'treebag')
summary(bagging_model_pro2)

predict_Bagging_Property2 = predict(bagging_model_pro2,newdata = test_main_Property2)
predict_Bagging_Property2
predict_Bagging_Property2 = ifelse(predict_Bagging_Property2>=0.5,1,0)
predict_Bagging_Property2

confusionMatrix(factor(predict_Bagging_Property2,levels = 0:1),factor(test_main_Property2$Property2,levels = 0:1))




#Random Forest


randomF_model_pro2 = randomForest(data=train_main_Property2,Property2~.)
randomF_model_pro2

predict_RandomF_Property2 = predict(randomF_model_pro2,newdata = test_main_Property2)
predict_RandomF_Property2
predict_RandomF_Property2 = ifelse(predict_RandomF_Property2>=0.5,1,0)
predict_RandomF_Property2

confusionMatrix(factor(predict_RandomF_Property2,levels = 0:1),factor(test_main_Property2$Property2,levels = 0:1))


#Support Vector Regression


library(e1071)
svr_model_pro2 = svm(data=train_main_Property2,Property2~.)
svr_model_pro2

predict_svr_Property2 = predict(svr_model_pro2,newdata = test_main_Property2)
predict_svr_Property2
predict_svr_Property2 = ifelse(predict_svr_Property2>=0.5,1,0)
predict_svr_Property2

confusionMatrix(factor(predict_svr_Property2,levels = 0:1),factor(test_main_Property2$Property2,levels = 0:1))


# Gaussian Process Regression


library(kernlab)
gpr_model_pro2 = gausspr(data=train_main_Property2,Property2~.)

predict_gpr_Property2 = predict(gpr_model_pro2,newdata = test_main_Property2)
predict_gpr_Property2
predict_gpr_Property2 = ifelse(predict_gpr_Property2>=0.5,1,0)
predict_gpr_Property2

confusionMatrix(factor(predict_gpr_Property2,levels = 0:1),factor(test_main_Property2$Property2,levels = 0:1))



#Linear Regression for all data


linear_model_pro2 = lm(data=train_main_Property2,Property2~.)
linear_model_pro2
summary(linear_model_pro2)

predict_Linear_Property2 = predict(linear_model_pro2,newdata = test_main_Property2)
predict_Linear_Property2
predict_Linear_Property2 = ifelse(predict_Linear_Property2>=0.5,1,0)
predict_Linear_Property2

confusionMatrix(factor(predict_Linear_Property2,levels = 0:1),factor(test_main_Property2$Property2,levels = 0:1))



#to check accuracy for intial property 2 without NA values


lrmodel_property2_testing = lm(data=train_Property2,Property2~c(Feature9+Feature11+Feature12))
lrmodel_property2_testing
summary(lrmodel_property2_testing)

predict_Linear_Property2_testing = predict(lrmodel_property2_testing,newdata = test_Property2)
predict_Linear_Property2_testing
predict_Linear_Property2_testing = ifelse(predict_Linear_Property2_testing>=0.5,1,0)
predict_Linear_Property2_testing

confusionMatrix(factor(predict_Linear_Property2_testing,levels = 0:1),factor(test_Property2$Property2,levels = 0:1))