# kNN classification
# multiclass classification
# dataset: Mobile_data.csv

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import neighbors
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import preprocessing
# visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

# read file

data = pd.read_csv("Mobile_data.csv")

data.head()
data.tail()
data.dtypes
data.shape
data.info()
data.describe()

# check the distribution of the y-variable
data.price_range.value_counts()

# function to plot the histogram, correlation matrix, boxplot based on the chart-type'
### list of functions
def splitcols(data):
    nc=data.select_dtypes(exclude='object').columns.values
    fc=data.select_dtypes(include='object').columns.values
    
    return(nc,fc)

def plotdata(data,nc,ctype):
    if ctype not in ['h','c','b']:
        msg='Invalid Chart Type specified'
        return(msg)
    
    if ctype=='c':
        cor = data[nc].corr()
        cor = np.tril(cor)
        sns.heatmap(cor,vmin=-1,vmax=1,xticklabels=nc,
                    yticklabels=nc,square=False,annot=True,linewidths=1)
    else:
        COLS = 2
        ROWS = np.ceil(len(nc)/COLS)
        POS = 1
        
        fig = plt.figure() # outer plot
        for c in nc:
            fig.add_subplot(ROWS,COLS,POS)
            if ctype=='b':
                sns.boxplot(data[c],color='yellow')
            else:
                sns.distplot(data[c],bins=20,color='green')
            
            POS+=1
    return(1)

# split the dataset into train and test in the ratio (default=70/30)

def splitdata(data,y,ratio=0.3):
    
    trainx,testx,trainy,testy = train_test_split(data.drop(y,1),
                                                 data[y],
                                                 test_size = ratio )
    
    return(trainx,testx,trainy,testy)

#Split the data in to numericand objects
nc,fc = splitcols(data)
print(nc)
print(fc)

#Check of correlation
plotdata(data,nc,'c') 
#Check for Outlier
plotdata(data,nc,'b') #Data has no outliers


#Check for the Nulls

data.isnull().sum() #No null found

#check for the zeros

data[nc][data[nc]==0].count() 

#Screen Width cannot be zero so replaced all the values with random of min and max values others are valid zeros

data.sc_w.describe()
data[data.sc_w == 0] = np.random.randint(1,18)



#=======================================================================
# standardize the data (only features have to be standardized)
# StandardScaler
# MinMaxScaler

# make a copy of the dataset
data_std = data.copy()

ss = preprocessing.StandardScaler()
sv = ss.fit_transform(data_std.iloc[:,:])
data_std.iloc[:,:] = sv

# restore the original Y-value in the data_std
data_std.price_range = data.price_range

# compare the actual and transformed data
data.head()
data_std.head()

# split the data into train/test
trainx,testx,trainy,testy=train_test_split(data_std.drop('price_range',1),
                                           data_std.price_range,
                                           test_size=0.3)


trainx.shape,trainy.shape
testx.shape,testy.shape

# cross-validation to determine the best K
cv_accuracy = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx,trainy,cv=10,scoring='accuracy')
    cv_accuracy.append(scores.mean() )

print(cv_accuracy) 

bestK = n_list[cv_accuracy.index(max(cv_accuracy))]
print("best K = ", bestK)

# plot the Accuracy vs Neighbours to determine the best K
plt.plot(n_list,cv_accuracy)
plt.xlabel("Neighbours")
plt.ylabel("Accyuracy")
plt.title("Accuracy - Neighbours")

# build the model using the best K
m1 = neighbors.KNeighborsClassifier(n_neighbors=bestK).fit(trainx,trainy)
# metric = "manhattan"
# predict on test data
p1 = m1.predict(testx)

# confusion matrix and classification report
df1=pd.DataFrame({'actual':testy,'predicted':p1})

pd.crosstab(df1.actual,df1.predicted,margins=True)
print(classification_report(df1.actual,df1.predicted))


#========================================================================
#Build model using MinMax scalar

data_1 = data.copy()

# standardize the data (only features have to be standardized)
# StandardScaler
# MinMaxScaler

# make a copy of the dataset
data_std1 = data_1.copy()

mm = preprocessing.MinMaxScaler()
sv = mm.fit_transform(data_std1.iloc[:,:])
data_std1.iloc[:,:] = sv

# restore the original Y-value in the data_std
data_std1.price_range = data_1.price_range

# compare the actual and transformed data
data_1.head()
data_std1.head()

# split the data into train/test
trainx1,testx1,trainy1,testy1=train_test_split(data_std1.drop('price_range',1),
                                           data_std1.price_range,
                                           test_size=0.3)


trainx1.shape,trainy1.shape
testx1.shape,testy1.shape

# cross-validation to determine the best K
cv_accuracy_1 = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx1,trainy1,cv=10,scoring='accuracy')
    cv_accuracy_1.append(scores.mean() )


print(cv_accuracy_1) 

bestK = n_list[cv_accuracy_1.index(max(cv_accuracy_1))]
print("best K = ", bestK)

# plot the Accuracy vs Neighbours to determine the best K
plt.plot(n_list,cv_accuracy_1)
plt.xlabel("Neighbours")
plt.ylabel("Accyuracy")
plt.title("Accuracy - Neighbours")

# build the model using the best K
m2 = neighbors.KNeighborsClassifier(n_neighbors=bestK).fit(trainx1,trainy1)
# metric = "manhattan"
# predict on test data
p2 = m2.predict(testx1)

# confusion matrix and classification report
df2=pd.DataFrame({'actual':testy1,'predicted':p2})

pd.crosstab(df2.actual,df2.predicted,margins=True)
print(classification_report(df2.actual,df2.predicted))


#======================================================================================================

#Build model using Feature selection (M3)

#Build the function for feature selection
#Returns the scores of all the features of train data set
#Input(train(x&y))


from sklearn.feature_selection import f_classif
def bestFeatures(trainx1,trainy1):
    features = trainx1.columns
    
    fscore,pval = f_classif(trainx1,trainy1)
    
    df = pd.DataFrame({"feature":features,"fscore":fscore,"pval":pval})
    df = df.sort_values("fscore",ascending = False)
    return(df)

#-------------------------------------------------------------------
bestFeatures(trainx1,trainy1)


#copy of the data
data_2 = data_std1.copy()

#Non-significanT Features
data_2.columns

drp_cls = ["fc","px_height","sc_w","int_memory","pc","talk_time","sc_h"]

data_2.drop(columns = drp_cls,inplace=True)
data_2.columns

#Build model 3 (M3) using data_2
data_2.columns

# split data
trainx2,testx2,trainy2,testy2 = train_test_split(data_2.drop('price_range',1),
                                             data_2.price_range,
                                             test_size=0.25)

print(trainx2.shape,trainy2.shape)
print(testx2.shape,testy2.shape)


# cross-validation to determine the best K
cv_accuracy_2 = []

n_list = np.arange(3,12,2); n_list

for n in n_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(model,trainx2,trainy2,cv=10,scoring='accuracy')
    cv_accuracy_2.append(scores.mean() )

print(cv_accuracy_2) 

bestK = n_list[cv_accuracy_2.index(max(cv_accuracy_2))]
print("best K = ", bestK)

# plot the Accuracy vs Neighbours to determine the best K
plt.plot(n_list,cv_accuracy_2)
plt.xlabel("Neighbours")
plt.ylabel("Accyuracy")
plt.title("Accuracy - Neighbours")

# build the model using the best K
m3 = neighbors.KNeighborsClassifier(n_neighbors=bestK).fit(trainx2,trainy2)
# metric = "manhattan"
# predict on test data
p3 = m3.predict(testx2)

# confusion matrix and classification report
df2=pd.DataFrame({'actual':testy2,'predicted':p3})

pd.crosstab(df2.actual,df2.predicted,margins=True)
print(classification_report(df2.actual,df2.predicted))

#=============================================================================
        '''Conclusion : - 

            ##  Model M1  using Z Score standerdisation 
                we got 
          precision    recall  f1-score   support

           0       0.76      0.79      0.77       150
           1       0.54      0.51      0.53       133
           2       0.49      0.59      0.53       117
           3       0.83      0.69      0.75       139
          12       1.00      1.00      1.00        61

    accuracy                           0.69       600
   macro avg       0.72      0.72      0.72       600
weighted avg       0.70      0.69      0.69       600

          

          ##  Model M2  using MinMax Scalar
              we got 
            precision    recall  f1-score   support

           0       0.79      0.80      0.80       142
           1       0.50      0.53      0.51       129
           2       0.53      0.60      0.56       131
           3       0.88      0.73      0.79       146
          12       1.00      1.00      1.00        52

    accuracy                           0.70       600
   macro avg       0.74      0.73      0.73       600
weighted avg       0.71      0.70      0.70       600

          ##  Model M3  using Feature selection
              we got 

         precision    recall  f1-score   support

           0       0.92      0.92      0.92       122
           1       0.82      0.78      0.80       121
           2       0.67      0.79      0.72        95
           3       0.91      0.83      0.87       119
          12       1.00      1.00      1.00        43

    accuracy                           0.85       500
   macro avg       0.86      0.86      0.86       500
weighted avg       0.85      0.85      0.85       500   


#Split ratio take for each of the model is 70/30
#M1 = Z score standerdise
#M2 = MinMax Standerdise
#M3 = Feature selection
So the bet model will be model M3 because it has higer accuracya(85%) and good clases distribution
'''


















