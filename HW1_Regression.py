import math
from scipy.stats import t
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


#Read in CSV to dataframe, and split dataframe into test data (~20%) and train data (~80%)
df=pd.read_csv('/Users/emax/Downloads/mtcars.csv')
np.random.seed(3)
msk = np.random.rand(len(df)) < 0.8

train_data=df[msk]
test_data=df[~msk]

#Save data for training in 2D arrays
Y_train=[[val] for val in train_data['mpg'].values]
X_train=[[val] for val in train_data['cyl'].values]

#Save data for testing in 1D arrays
Y_test=test_data['mpg'].values
X_test=test_data['cyl'].values

#Train linear model on training data
reg=linear_model.LinearRegression(fit_intercept=True)
reg.fit(X_train,Y_train)

#Save and print params and save model as function
beta0=reg.intercept_[0]
beta1=reg.coef_[0][0]
print("Y~ "+str(round(beta0,2))+str(round(beta1,2))+"* x")

def y(x):
    return(beta0+beta1*x)

#Apply model to training data
X_train_plot=[val[0] for val in X_train]
Y_train_pred=[y(x) for x in X_train_plot]
Y_train_true=[val[0] for val in Y_train]
    
#Plot model and data
plt.plot(X_train_plot, Y_train_pred, color='blue', linewidth=2, markersize=1.5)
plt.scatter(X_train_plot,Y_train_true, s=3,c='green')
    
mse=metrics.mean_squared_error(Y_train_pred,Y_train_true)
        
patch=matplotlib.patches.Patch(color='Red', label='Mean Squared Error: '+str(round(mse,2)))
plt.legend(handles=[patch])
    
plt.xlabel('cyl')
plt.ylabel('mpg')
plt.show()

#Apply model to testing data
X_test_plot=X_test
Y_test_pred=[y(x) for x in X_test_plot]
Y_test_true=Y_test
    
#Plot model and data
plt.plot(X_test_plot, Y_test_pred, color='blue', linewidth=2, markersize=1.5)
plt.scatter(X_test_plot,Y_test_true, s=3,c='green')

mse=metrics.mean_squared_error(Y_test_pred,Y_test_true)

patch=matplotlib.patches.Patch(color='Red', label='Mean Squared Error: '+str(round(mse,2)))
plt.legend(handles=[patch])

plt.xlabel('cyl')
plt.ylabel('mpg')
plt.show()

#Using the same test/train split above, add a second feature
#(wt) to 2D array for training data and 2D array for testing
#data
X_train=train_data[['cyl','wt']].values
X_test=test_data[['cyl','wt']].values

#Train linear model on training data
reg=linear_model.LinearRegression(fit_intercept=True)
reg.fit(X_train,Y_train)

#Save and print params and save model as function
beta0=reg.intercept_[0]
beta1=reg.coef_[0][0]
beta2=reg.coef_[0][1]
def y(x1,x2):
    return(beta0+beta1*x1+beta2*x2)
print("Y~ "+str(round(beta0,2))+" "+str(round(beta1,2))+"*x1 "+str(round(beta2,2))+"* x2")

#Intuitive check for correlation of wt and cyl
Y_check=[val[1] for val in X_train]
X_check=[val[0] for val in X_train]

reg.fit([[x] for x in X_check],[[y] for y in Y_check])
beta0=reg.intercept_[0]
beta1=reg.coef_[0][0]

Y_check_pred=[beta0+beta1*x for x in X_check]

plt.scatter(X_check,Y_check, c='red', s=.8)
plt.plot(X_check,Y_check_pred,color='blue',linewidth=1, markersize=1)
plt.xlabel('# of Cylinders')
plt.ylabel('Weight')
plt.show()

#Apply model to testing data and training data and compute MSE
Y_train_pred=[y(val[0],val[1]) for val in X_train]
Y_train_true=Y_train
mse=metrics.mean_squared_error(Y_train_pred,Y_train_true)
print('Training MSE: '+str(round(mse,2)))

Y_test_pred=[y(val[0],val[1]) for val in X_test]
Y_test_true=Y_test
mse=metrics.mean_squared_error(Y_test_pred,Y_test_true)
print('Testing MSE: '+str(round(mse,2)))

#Add interaction term to training and testing data
X_train=[[val[0],val[1],val[0]*val[1]] for val in X_train]
X_test=[[val[0],val[1],val[0]*val[1]] for val in X_test]

#Train linear model on training data
reg=linear_model.LinearRegression(fit_intercept=True)
reg.fit(X_train,Y_train)

#Save and print params and save model as function
beta0=reg.intercept_[0]
beta1=reg.coef_[0][0]
beta2=reg.coef_[0][1]
beta3=reg.coef_[0][2]

def y(x1,x2):
    return(beta0+beta1*x1+beta2*x2+beta3*x1*x2)
    
print("Y~ "+str(round(beta0,2))+" "+str(round(beta1,2))+"*x1 "+str(round(beta2,2))+"* x2 "+str(round(beta3,2))+"* x1x2")


#Apply model to testing data and training data and compute MSE
Y_train_pred=[y(val[0],val[1]) for val in X_train]
Y_train_true=Y_train
mse=metrics.mean_squared_error(Y_train_pred,Y_train_true)
print('Training MSE: '+str(round(mse,2)))

Y_test_pred=[y(val[0],val[1]) for val in X_test]
Y_test_true=Y_test
mse=metrics.mean_squared_error(Y_test_pred,Y_test_true)
print('Testing MSE: '+str(round(mse,2)))

#Read in CSV to dataframe
df=pd.read_csv('/Users/emax/Downloads/wage_data.csv')
#Add CSV column for age^2
df['age_sqr']=df.apply(lambda row: row['age']**2, axis=1)

#Split dataframe into test data and train data
np.random.seed(3)
msk = np.random.rand(len(df)) < 0.8
    
train_data=df[msk]
test_data=df[~msk]

#Format training and testing data
X_train=train_data[['age','age_sqr']].values
Y_train=[[val] for val in train_data['wage'].values]

X_test=test_data['age'].values
Y_test=test_data['wage'].values

#Train linear model on training data
reg=linear_model.LinearRegression(fit_intercept=True)
reg.fit(X_train,Y_train)
    
#Save and print params and save model as function
beta0=reg.intercept_[0]
beta1=reg.coef_[0][0]
beta2=reg.coef_[0][1]
print("Y~ "+str(round(beta0,2))+" "+str(round(beta1,2))+"*x "+str(round(beta2,2))+"* x^2 ")
def y(x):
    return(beta0+beta1*x+beta2*x**2)

#Get predictions from training data
Y_train_pred=[y(val[0]) for val in X_train]

#Get degrees of freedom
deg_f=len(Y_train_pred)-3

#Compute MSres from training data
pred_true_df=pd.DataFrame({'Pred.':Y_train_pred, 'True':[val[0] for val in Y_train]})
pred_true_df['Resid_sqr']=pred_true_df.apply(lambda row: (row['Pred.']-row['True'])**2, axis=1)
    
RSS=sum(pred_true_df['Resid_sqr'])
MSres=RSS/deg_f

#Get tc critical value from t distribution
t_c=t.ppf(.025, df=deg_f)

#Save training data for plotting             
Y_train_plot=[val[0] for val in Y_train]
X_train_plot=[val[0] for val in X_train]
    
#Propend column of 1s to data array and interpret as matrix
X_train=np.asmatrix([[1,val[0],val[1]] for val in X_train])
X_train_T=X_train.transpose()
C=np.dot(X_train_T,X_train).I

#Upper confidence window for y
def y_up(x):
    A=np.asmatrix([1,x,x**2])
    se=math.sqrt(MSres*np.dot(np.dot(A,C),A.transpose()))
    return(y(x)+t_c*se)
#Lower confidence window for y
def y_down(x):
    A=np.asmatrix([1,x,x**2])
    se=math.sqrt(MSres*np.dot(np.dot(A,C),A.transpose()))
    return(y(x)-t_c*se)

#Define range for function
X_fine=[i for i in range(min(min(X_test),min(X_train_plot)),max(max(X_test),max(X_train_plot)))]
Y_fine=[y(x) for x in X_fine]
Y_lower=[y_down(x) for x in X_fine]
Y_upper=[y_up(x) for x in X_fine]

#Plot it with confidence bounds
plt.plot(X_fine,Y_fine,color='blue',linewidth=1, markersize=.25)
plt.plot(X_fine,Y_lower,color='green',linewidth=1, markersize=.25)
plt.plot(X_fine,Y_upper,color='orange',linewidth=1, markersize=.25)
plt.fill_between(X_fine,Y_lower,Y_upper,facecolor='grey', alpha=0.3)

#Plot rest of data
plt.scatter(X_test,Y_test, c='red', s=.15)
plt.scatter(X_train_plot,Y_train_plot, c='purple', s=.15)
plt.xlabel('Age (yrs)')
plt.ylabel('Wage (1000s USD)')

plt.show()
