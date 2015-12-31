#set a seed
set.seed(1234)
#read data from the given CSV file
wine_data=read.csv("D6 Wine quality.csv")

#check for missing values
sum(is.na(wine_data))

#split data into training & testing subsets, in the ratio 70:30
grp=rbinom(nrow(wine_data),1,0.7)
training=wine_data[grp==1,]
testing=wine_data[grp==0,]

summary(training)       #summarize to find range of values of all variables
                        #shows that variables are of different scales, & need to be normalized
cor(training)           #analyse correlations between different variables
                        #shows that density is correlated with alcohol & residual sugar, & hence can be dropped 
table(training$quality) #find the distinct values/levels for wine quality

#plot boxplots to see if any variables can be used to distinguish different levels of wine quality
#save images in jpeg files
for(i in 1:11)
{
  filename=paste("boxplot_",i,".jpeg",sep="")
  
  jpeg(filename)
  boxplot(training[,i]~training[,12])
  dev.off()
}
#no concrete findings from the plots. (This step can be avoided)

#convert training & testing sets(data frames) into matrices. 
#remove 8th column(density) as it is correlated, & 12 column(quality) as it is the output
training_matrix=as.matrix(cbind(training[,-c(8,12)]))
testing_matrix=as.matrix(cbind(testing[,-c(8,12)]))

#do mu-sigma normalization on training_matrix, save result as training_norm
training_norm<-training_matrix
mu<-as.matrix(colMeans(training_matrix))
sigma<-as.matrix(apply(training_matrix,2,sd))
for (i in 1:nrow(training_matrix))
  for(j in 1:ncol(training_matrix))
    training_norm[i,j]<-(training_matrix[i,j]-mu[j])/sigma[j]

#add a column of 1's to training norm. (intercept column)
a=rep(1,nrow(training_matrix))
training_norm<-cbind(a,training_norm)

#convert quality variable into matrix y of 7 columns.
#if quality value is k, kth column in y will be 1, all other columns 0
y=matrix(0,nrow(training),7)
for (i in 1:nrow(training))
{
  for (j in 1:7)
  {
    val=as.numeric(training[i,12])
    if(val==j+2)
      y[i,j]=1
    
  }
}

#sigmoid function definition
sigmoid<-function(x)
{
  g=1/(1+ exp(-x))
}

#cost function definition
cost<-function(x,y,theta)
{
  #x=[m*n]
  #y=[m*levels]
  #theta=[n*levels]
  m=nrow(x)
  n=ncol(x)
  levels=ncol(theta)
  J=0
  grad=matrix(0,n,levels) #[n*levels]
  C=x%*%theta             #[m*levels]
  c1=(-log(sigmoid(C)))   #[m*levels]
  c2=(log(1-sigmoid(C)))  #[m*levels]
  
  J=sum(c1*y - c2*(1-y))/m
  grad=t(t(sigmoid(C)-y) %*% x)
  D=rep(J,levels)         #[1*levels]
  grad=rbind(D,grad)      #[(n+1)*levels]
  
  #we need to return cost J (single numeric value), & gradient grad (matrix [n*levels])
  #since a function can only have 1 return type, we convert J into a row vector of size 1*levels
  #& append it as the 1st row of grad matrix. (grad is now of size [(n+1)*levels])
  
  return(grad)
}

#gradient descent function to optimize theta
grad_desc<-function(alpha,theta,x,y,iter)
{
  J=rep(0,iter)
  m=nrow(x)
  
  for (i in 1:iter)
  {
    grad=matrix(0,nrow(theta)+1,ncol(theta))
    grad=cost(x,y,theta)
        
    J[i]=grad[1,1]      #extract cost value J from grad 
    grad=grad[-1,]      #remove out 1st row from grad matrix, as it contains the cost
    
    grad=grad * (alpha/m)
    theta=theta-grad
      
  }
 
  plot(J,pch=19,cex=0.6) #checking for convergence of gradient descent
  return (theta)
}

#one-vs-all classification function
#given x & y, assigns an initial Theta matrix.
#optimizes the Theta by calling gradient descent,& returns the final Theta
onevsall<-function(x,y)
{
  n=ncol(x)
  m=nrow(x)
  classes=ncol(y)
  
  Theta=matrix(0,n,classes)
  for(i in 1:n)
  {
    for(j in 1:classes)
    {
      Theta[i,j]=runif(1,0,1) #randomly initializing values for Theta
    }
  }

  alpha=1         #learning rate
  iter=100        #number of iterations for gradient descent
  #tweak these values if graph is non-converging
  Theta1=grad_desc(alpha,Theta,x,y,iter)
  return (Theta1)
}

#call onevsall function passing training_norm & y
Theta_optimal=onevsall(training_norm,y)

#use the returned Theta_optimal to make predictions
P1=sigmoid(training_norm %*% Theta_optimal)

#P1= matrix [m * levels]
#predicted class for each row= column with highest value for that particular row of P1
predictions=rep(0,nrow(P1))
for (i in 1:nrow(P1))
{
  m=1
  for (j in 2:ncol(P1))
  {
    if (P1[i,j]>(P1[i,m]))
      m=j
  }
  predictions[i]=m+2        #m+2 is done bcoz array index is from 1 to 7, whereas class labels are from 3 to 9
}

#normalize the testing set, using the mu & sigma calculated earlier
testing_norm<-testing_matrix
for (i in 1:nrow(testing_matrix))
  for(j in 1:ncol(testing_matrix))
    testing_norm[i,j]<-(testing_matrix[i,j]-mu[j])/sigma[j]

#add intercept column
a=rep(1,nrow(testing_matrix))
testing_norm=cbind(a,testing_norm)

#obtain predictions on testing set, folloing same approach as before
P2=sigmoid(testing_norm %*% Theta_optimal)
predictions2=rep(0,nrow(P2))
for (i in 1:nrow(P2))
{
  m=1
  for (j in 2:ncol(P2))
  {
    if (P2[i,j]>(P2[i,m]))
      m=j
  }
  predictions2[i]=m+2
}

#tabulate results. our predictions are along the rows, actual values (ground truth) along columns
table(predictions,training$quality)
table(predictions2,testing$quality)
#you should expect to see an accuracy of around 53-55% on both training & testing sets

#now we compare our model's accuracy against other models. We do the comarison only on the testing set
#most inbuilt models need class labels to be in factor form instead of numeric, so do the conversion
training$quality=as.factor(training$quality)
testing$quality=as.factor(testing$quality)

#hack model, which predicts the most commonly occuring value (6) as wine quality output, irrespective of input 
hack_pred=rep(6,nrow(testing))
table(hack_pred,testing$quality)
#accuracy of 44-46%. our model gives better results than this. [thank goodness! :)]

#decision tree model, using library- tree
library(tree)
model1=tree(quality~.,data=training[,-8])      #quality is output, all others are input variables, 
                                               #with data being training set excluding 8th column(density)
p1=predict(model1,testing[,-c(8,12)],type="class")
table(p1,testing$quality)
plot(model1)
text(model1)
#accuracy of around 49-52%
#wow. our model actually does better than this one too :)

#ensemble learning model, using random forests
library(randomForest)
model2=randomForest(quality~.,data=training[,-8],ntree=150,importance=TRUE) 
#generates a forest consisting of 150 trees
p2=predict(model2,testing[,-c(8,12)])
table(p2,testing$quality)
#accuracy of around 66-69%
#we are nowhere close to this, but we'll keep trying to improve & eventually get there.
