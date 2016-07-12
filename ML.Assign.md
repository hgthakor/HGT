# MLAssign.Rmd
HGTHAKOR  
July 12, 2016  

## Predicting the manner in which they did the exercise.

### Introduction
This human activity recognition research focus on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time. However, the "how (well)" investigation has only received little attention so far. This analysis tries to answer this question.

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). It is possible to collect a large amount of data about personal activity using devices such as Jawbone Up, Nike FuelBand, and Fitbit. People regularly quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal is to use (given) data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

In this analysis we will first define quality of execution and investigate three aspects that pertain to qualitative activity recognition: the problem of specifying correct execution, the automatic and robust detection of execution mistakes, and how to provide feedback on the quality of execution to the user. 

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. It was made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

### Data Loading


```r
library(dplyr)
```

```
## Warning: package 'dplyr' was built under R version 3.2.5
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.5
```

```r
library(RColorBrewer)
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.2.5
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
library(knitr)
```

```
## Warning: package 'knitr' was built under R version 3.2.5
```

```r
library(AppliedPredictiveModeling)
```

```
## Warning: package 'AppliedPredictiveModeling' was built under R version
## 3.2.5
```

Also load

```r
library(ggplot2)
library(lubridate)
```

```
## Warning: package 'lubridate' was built under R version 3.2.5
```

```
## 
## Attaching package: 'lubridate'
```

```
## The following object is masked from 'package:base':
## 
##     date
```



```r
cache = TRUE
include = TRUE
eval=TRUE
collape = TRUE
fig.width = 8
fig.height = 6
echo = TRUE 
warning = FALSE
Results ='markup'
options(scipen = 1) 
```

###Loading Data:

```r
training<- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""), header=TRUE)
testing<- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""), header=TRUE)
```



```r
str(training, list.len=10)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "2/12/2011 13:32",..: 15 15 15 15 15 15 15 15 15 15 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##   [list output truncated]
```


```r
dim(training)
```

```
## [1] 19622   160
```


```r
colnames_train <- colnames(training)
colnames_test <- colnames(testing)
```
###Verify that the column names (excluding classe and problem_id) are identical in the training and test set

```r
all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_train)-1])
```

```
## [1] TRUE
```

###Transforming data: Converting date and adding new variable (Day)

```r
training$cvtd_timestamp<- as.Date(training$cvtd_timestamp, format = "%m/%d/%Y %H:%M")
training$Day<-factor(weekdays(training$cvtd_timestamp))
```

### Exploratory Data Analysis

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```


```r
prop.table(table(training$classe))
```

```
## 
##         A         B         C         D         E 
## 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243
```


```r
prop.table(table(training$user_name))
```

```
## 
##    adelmo  carlitos   charles    eurico    jeremy     pedro 
## 0.1983488 0.1585975 0.1802059 0.1564570 0.1733768 0.1330140
```


```r
prop.table(table(training$user_name,training$classe),1)
```

```
##           
##                    A         B         C         D         E
##   adelmo   0.2993320 0.1993834 0.1927030 0.1323227 0.1762590
##   carlitos 0.2679949 0.2217224 0.1584190 0.1561697 0.1956941
##   charles  0.2542421 0.2106900 0.1524321 0.1815611 0.2010747
##   eurico   0.2817590 0.1928339 0.1592834 0.1895765 0.1765472
##   jeremy   0.3459730 0.1437390 0.1916520 0.1534392 0.1651969
##   pedro    0.2452107 0.1934866 0.1911877 0.1796935 0.1904215
```


```r
prop.table(table(training$user_name,training$classe),2)
```

```
##           
##                    A         B         C         D         E
##   adelmo   0.2087814 0.2043719 0.2191701 0.1601368 0.1901857
##   carlitos 0.1494624 0.1817224 0.1440678 0.1511194 0.1688384
##   charles  0.1611111 0.1962075 0.1575102 0.1996269 0.1971167
##   eurico   0.1550179 0.1559126 0.1428989 0.1809701 0.1502634
##   jeremy   0.2109319 0.1287859 0.1905319 0.1623134 0.1558082
##   pedro    0.1146953 0.1329997 0.1458212 0.1458333 0.1377876
```


```r
prop.table(table(training$classe, training$Day),1)
```

```
##    
##      Saturday  Thursday
##   A 0.5833804 0.4166196
##   B 0.5600147 0.4399853
##   C 0.5651030 0.4348970
##   D 0.5478220 0.4521780
##   E 0.5581302 0.4418698
```


```r
qplot(x=Day, fill=classe, data = training)
```

![](ML.Assign_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

### Inferences from exploratory data analysis
The stake graph and the analysis shows that: 
1. The most frequently used activity (28.5%) is Class-A activity. It is most frequently used by user-Jeremy (34.6%)
2.Adelmo is the most frequent user of across acitivities (19.8%).  but he uses Class "C" activity most frequently among all the users.
3.Majority of the actitivies happened during Saturdays and  the most frequently used activites are Classes A and B.
4.The most class A activity user is Jeremy and he is least frequent user of class B.


Partitioning the training set

```r
inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
dim(myTraining); dim(myTesting)
```

```
## [1] 11776   161
```

```
## [1] 7846  161
```

### Data Cleaning:


```r
myDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)
```


```r
myNZVvars <- names(myTraining) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt",
"kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt",
"max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm",
"var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm",
"stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm",
"kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm",
"max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm",
"kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell",
"skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell",
"amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm",
"skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm",
"max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm",
"amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm",
"avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm",
"stddev_yaw_forearm", "var_yaw_forearm")
myTraining <- myTraining[!myNZVvars]
```


```r
dim(myTraining)
```

```
## [1] 11776   101
```
Removing first ID variable so that it does not interfer with ML Algorithms:


```r
myTraining <- myTraining[c(-1)]
```
Cleaning Variables with too many NAs. For Variables that have more than a 60% threshold of NA's we will leave them out:

```r
trainingV3 <- myTraining #creating another subset to iterate in loop
for(i in 1:length(myTraining)) { #for every column in the training dataset
        if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .6 ) { #if n?? NAs > 60% of total observations
        for(j in 1:length(trainingV3)) {
            if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) ==1)  { #if the columns are the same:
                trainingV3 <- trainingV3[ , -j] #Remove that column
            }   
        } 
    }
}
```
Checking the new N

```r
dim(trainingV3)
```

```
## [1] 11776    59
```
Seting back to our set

```r
myTraining <- trainingV3
rm(trainingV3)
```

Now  we will do the same 3 transformations steps with myTesting and testing data sets.

```r
clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -58])  # remove the classe column
myTesting <- myTesting[clean1]         # allow only variables in myTesting that are also in myTraining
```


```r
dim(myTesting)
```

```
## [1] 7846   59
```


For ensuring proper functioning of Decision Trees and especially RandomForest Algorithm with the Test data set (data set provided), we need to coerce the data into the same type.

Simple smart ass technique used to make sure Coertion really worked, 


Now using ML algorithms for prediction: Decision Tree

```r
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")
```

Now run the following command to view the decision tree with fancy 

```r
fancyRpartPlot(modFitA1)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![](ML.Assign_files/figure-html/unnamed-chunk-28-1.png)<!-- -->

For predicting: 

```r
predictionsA1 <- predict(modFitA1, myTesting, type = "class")
```

Now Using confusion Matrix to test results:

```r
confusionMatrix(predictionsA1, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2039   51    3    0   21
##          B   48 1205   59  104   62
##          C   70  154 1229  150   39
##          D   73   97   77  969   78
##          E    2   11    0   63 1242
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8519          
##                  95% CI : (0.8438, 0.8597)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8131          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9135   0.7938   0.8984   0.7535   0.8613
## Specificity            0.9866   0.9569   0.9362   0.9505   0.9881
## Pos Pred Value         0.9645   0.8153   0.7485   0.7488   0.9423
## Neg Pred Value         0.9663   0.9508   0.9776   0.9516   0.9694
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2599   0.1536   0.1566   0.1235   0.1583
## Detection Prevalence   0.2694   0.1884   0.2093   0.1649   0.1680
## Balanced Accuracy      0.9501   0.8753   0.9173   0.8520   0.9247
```



```r
predictionsA1 <- predict(modFitA1, myTesting, type = "class")
cmtree <- confusionMatrix(predictionsA1, myTesting$classe)
cmtree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2039   51    3    0   21
##          B   48 1205   59  104   62
##          C   70  154 1229  150   39
##          D   73   97   77  969   78
##          E    2   11    0   63 1242
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8519          
##                  95% CI : (0.8438, 0.8597)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8131          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9135   0.7938   0.8984   0.7535   0.8613
## Specificity            0.9866   0.9569   0.9362   0.9505   0.9881
## Pos Pred Value         0.9645   0.8153   0.7485   0.7488   0.9423
## Neg Pred Value         0.9663   0.9508   0.9776   0.9516   0.9694
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2599   0.1536   0.1566   0.1235   0.1583
## Detection Prevalence   0.2694   0.1884   0.2093   0.1649   0.1680
## Balanced Accuracy      0.9501   0.8753   0.9173   0.8520   0.9247
```


```r
plot(cmtree$table, col = cmtree$byClass, main = paste("Decision Tree Confusion Matrix: Accuracy =", round(cmtree$overall['Accuracy'], 4)))
```

![](ML.Assign_files/figure-html/unnamed-chunk-32-1.png)<!-- -->

Prediction with randomForest

As the Knitting HTML file was not able to read this code it has been removed.
