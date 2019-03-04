###############################################################################
##################################
         #data cleaning
##################################
# import data
rm(list=ls())
library(tidyverse)
dataset<-read.csv("breast-cancer.csv")
glimpse(dataset)
data<-dataset[,-c(1,dim(dataset)[2])]
data$diagnosis<-as.numeric(data$diagnosis)
data$diagnosis[which(data$diagnosis==2)]<-0
# look through label, plot
summary(data)
dis<-as.data.frame(prop.table(table(dataset$diagnosis)))
dis$Freq<-round(dis$Freq,digits = 3)
quartz()
ggplot(dis,aes(Var1,Freq)) +
  geom_bar(stat="identity",width = 0.3) +
  geom_text(aes(label = Freq, vjust = -0.4, hjust = 0.5))+
  xlab("Label") + ylab("") +
  ggtitle("Distribution of Label") +
  theme(plot.title = element_text(hjust = 0.5),legend.position="none") 
# check if there is any missing value
map_int(data, function(x) sum(is.na(x)))

#chech correlations
library(corrplot)
corrplot(corr=cor(data[,-1]), order = "hclust",tl.cex = 1, addrect = 8,tl.col=1)

#positive 
quartz()
ggplot(data,aes(perimeter_mean,area_mean,color=as.factor(diagnosis)))+
  geom_point() + ggtitle("positive correlation variables") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("perimeter_mean") + ylab("area_mean") +
  scale_colour_discrete(name = "Label",labels = c("Malignant","Benign")) 

#invverse
quartz()
ggplot(data,aes(fractal_dimension_mean,radius_mean,color=as.factor(diagnosis)))+
  geom_point() + ggtitle("negative correlation variables") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("fractal_dimension_mean") + ylab("radius_mean") +
  scale_colour_discrete(name = "Label",labels = c("Malignant","Benign")) 

#uncorrelative
quartz()
ggplot(data,aes(log(symmetry_se),log(concavity_worst),color=as.factor(diagnosis)))+
  geom_point() + ggtitle("uncorrelation variables") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("symmetry_se") + ylab("concavity_worst") +
  scale_colour_discrete(name = "Label",labels = c("Malignant","Benign")) 


# split
library(caTools)
set.seed(1)
split = sample.split(Y = data$diagnosis, SplitRatio = 0.8)
train = subset(data, split == TRUE)
test = subset(data, split == FALSE)
x.train<-train[,-1]
y.train<-train[,1]
x.test<-test[,-1]
y.test<-test[,1]
x.train <- as.matrix(x.train)
x.test <- as.matrix(x.test)


###############################################################################
#############################
# random forest
#############################
# rank


library(randomForest)
forest<-randomForest(as.factor(diagnosis)~.,data=train)
vif = as.data.frame(importance(forest)/sum(importance(forest)))
v<-cbind(vif,as.data.frame(rownames(vif)))
colnames(v)<-c('VIF','name')
quartz()
ggplot(v,aes(x=reorder(name,-VIF),y=VIF)) +
  geom_bar(stat="identity") +
  geom_abline(intercept = 0.01, color = "red", slope = 0)+
  xlab("variables") + ylab("importance") +
  ggtitle("Feature Importance Proportion") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle=90),legend.position="none") +
  scale_y_continuous(breaks = seq(0,0.15,0.03))

rankv<-v[order(v$VIF,decreasing = TRUE),]
#l<-c('diagnosis',rownames(rankv[which(rankv$VIF>=0.01),]))

#rf<-randomForest(as.factor(diagnosis)~.,data=train[,l])
forestt<-predict(forest,x.train)
ftest<-predict(rf,x.test)
forest.train<-mean(forestt!=y.train)
forest.test<-mean((ftest!=y.test))
list(forest.train = forest.train,
     forest.test = forest.test)

library(caret)
as.table(confusionMatrix(forestt,as.factor(y.train)))
as.table(confusionMatrix(ftest,as.factor(y.test)))

#############################
# lasso
#############################
library(glmnet)
library(reshape)
lambda.grid <- 10^seq(4,-4,length=100)
lasso.mod=glmnet(x.train,y.train,alpha=1,lambda=lambda.grid)
set.seed(1)
coeff<-coef(lasso.mod)
quartz()
plot(coeff[2,],type = 'l',ylim = c(min(coeff[-1,]),max(coeff[-1,])))
for (i in 2:31) {
  lines(coeff[i,],col=i,type='l')
}

train.error <- rep(NA,100)
test.error<-rep(NA,100)
for(i in 1:100){
  lasso.pred <- predict(lasso.mod,s=lambda.grid[i],newx=x.train)
  test.pred<-predict(lasso.mod,s=lambda.grid[i],newx=x.test)
  train.error[i] <- mean((lasso.pred-y.train)^2)
  test.error[i]<-mean((test.pred-y.test)^2)
}
plt<-cbind(as.data.frame(lambda.grid),
      as.data.frame(test.error),as.data.frame(train.error))
lambda.lasso <- lambda.grid[which.min(test.error)]
quartz()
ggplot(data=melt(plt[,c(1,2,3)],id.vars = 1))+
  geom_line(aes(log(lambda.grid),value,col = variable))+
  ggtitle('Error change with different lambda')+ 
  ylab("Error") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_colour_discrete(name = 'Type',labels = c('Train','Test'))

  




###############################################################################
#############################
# pc regression
#############################
library(ggplot2)
pr.out<-prcomp(train[,-1],scale = TRUE, center = TRUE)
pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)
# loadngs
loadings<-loadings(pr.out)$PC1
pc1<-sort(loadings)
#plot for variance
#plot(pve, xlab="Principal Component", ylab="Proportion", 
#     ylim=c(0,1),type='b',main = "Proportion of Variance Explained")

# plot cumulative variance
pve.table<-tibble(x = seq(1,length(pve)),pve,cumsum(pve))
quartz()
ggplot(pve.table,aes(x,cumsum(pve))) + 
  geom_point() + geom_line() +
  geom_abline(intercept = 0.85, color = "red", slope = 0) +
  ggtitle("Cumulative variance") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("number of component") + ylab("cumulative variance")

# plot components
library(devtools)
install_github("vqv/ggbiplot")
library(ggbiplot)
quartz()
par(mfrow=c(1,2))
ggbiplot(pr.out,groups = as.factor(train$diagnosis),choices = c(1,2),ellipse = TRUE) +
  scale_colour_manual(name="Diagnosis", values= c("red3", "dark blue"))+
  ggtitle("Principal Component 1-2")+
  theme_minimal()+
  theme(legend.position = "bottom")
ggbiplot(pr.out,groups = as.factor(train$diagnosis),choices = c(3,4),ellipse = TRUE) +
  scale_colour_manual(name="Diagnosis", values= c("red3", "dark blue"))+
  ggtitle("Principal Component 3-4")+
  theme_minimal()+
  theme(legend.position = "bottom")

library(pls)
pcr.fit<-pcr(diagnosis~.,data=train,scale=TRUE,validation="none")
pcr.test<-predict(pcr.fit,x.test,ncomp=4)
pcr.train<-predict(pcr.fit,x.train,ncomp=4)

# roc 
pcr.train.error<-mean((pcr.train - y.train)>0.5)
pcr.test.error<-mean((pcr.test - y.test)>0.5)
list(pcr.train = pcr.train.error,
     pcr.test = pcr.test.error)

predict.train<-ifelse(pcr.train<0,0,ifelse(pcr.train<0.5,0,1))
predict.test<-ifelse(pcr.test<0,0,ifelse(pcr.test<0.5,0,1))

# confusion matrix

as.table(confusionMatrix(as.factor(predict.train),as.factor(y.train)))
as.table(confusionMatrix(as.factor(predict.test),as.factor(y.test)))
