setwd("D:/Users/tghunt/Dropbox/Learning/Coursera/Data Science/Pratical_Machine_Learning")

set.seed(12345)

library(data.table)
library(caret)
library(randomForest)
#library(plyr)

training <- data.table(read.csv("pml-training.csv"))

# set classe as a factor
training$classe <- as.factor(training$classe)

testing <- data.table(read.csv("pml-testing.csv"))

#Get the record numbers for each user
rcs_adelmo <- training[,which(user_name=="adelmo")]
rcs_pedro <- training[,which(user_name=="pedro")]
rcs_carlitos <- training[,which(user_name=="carlitos")]
rcs_charles <- training[,which(user_name=="charles")]
rcs_eurico <- training[,which(user_name=="eurico")]
rcs_jeremy <- training[,which(user_name=="jeremy")]

training_smaller <- training[,which( !grepl("X|num_window|min_|max_|avg_|var_|stddev_|kurtosis_|skewness_|amplitude_|new_window|user_name|timestamp",colnames(training) ) ), with=FALSE ]

testing_smaller <- testing[,which( !grepl("X|num_window|min_|max_|avg_|var_|stddev_|kurtosis_|skewness_|amplitude_|new_window|user_name|timestamp",colnames(testing) ) ), with=FALSE ]

#https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md

nsv <- nearZeroVar(training_smaller, saveMetrics = TRUE)


library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
#cluster <- 7
registerDoParallel(cluster)
nmLen <- length(names(training_smaller)) - 2
fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)

fitControl2 <- trainControl(method = "cv",
                           allowParallel = TRUE)

print(paste0("First fit start: ",Sys.time()))
if (file.exists("modFit1.RData")) {
  print("loading existing")
  load("modFit1.RData")
} else {
  print("Creating model")
  modFit1 <- train(classe~., method="rf", data = training_smaller, trControl = fitControl)
  save(modFit1, file="modFit1.RData")
}
print(paste0("First fit end: ",Sys.time()))
# approximately 10 minutes with an I7-3770K CPU

print(paste0("Second fit start: ",Sys.time()))
if (file.exists("modFit2.RData")) {
  print("loading existing")
  load("modFit2.RData")
} else {
  print("Creating model")
  modFit2 <- train(classe~., method="rf", data = training_smaller, preProcess=c("center"), trControl = fitControl)
  save(modFit2, file="modFit2.RData")
}
print(paste0("Second fit end: ",Sys.time()))
# approximately 10 minutes with an I7-3770K CPU

print(paste0("Third fit start: ",Sys.time()))
if (file.exists("modFit3.RData")) {
  print("loading existing")
  load("modFit3.RData")
} else {
  print("Creating model")
  modFit3 <- train(classe~., method="rf", data = training_smaller, preProcess=c("center", "scale"), trControl = fitControl)
  save(modFit3, file="modFit3.RData")
}
print(paste0("Third fit end: ",Sys.time()))
# approximately 10 minutes with an I7-3770K CPU

stopCluster(cluster)

predTrain1 <- predict(modFit1, training_smaller)
predTrain2 <- predict(modFit2, training_smaller)
predTrain3 <- predict(modFit3, training_smaller)

pred1 <- data.frame(predTrain1, classe=training_smaller$classe, agree=predTrain1 == training_smaller$classe)

pred2 <- data.frame(predTrain2, classe=training_smaller$classe, agree=predTrain2 == training_smaller$classe)

pred3 <- data.frame(predTrain3, classe=training_smaller$classe, agree=predTrain3 == training_smaller$classe)

cm1 <- confusionMatrix(predTrain1, training_smaller$classe)
cm2 <- confusionMatrix(predTrain2, training_smaller$classe)
cm3 <- confusionMatrix(predTrain3, training_smaller$classe)

print(cm1, digits = 3)
print(cm2, digits = 3)
print(cm3, digits = 3)

predTest <- predict(modFit1, testing_smaller)


missClass = function(values, pred1) {
  sum(pred1 != values)/length(values)
}
errRate = missClass(training_smaller$classe, predTrain1)



