import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #to split data
from sklearn.linear_model import LinearRegression #to create the model
from sklearn.metrics import mean_squared_error, r2_score #to obtain some numbers
#this is the ramdom seed to match the results, will discuss the impact of this
#choice later
import statsmodels.formula.api as smf
import statistics as st
from scipy.stats import norm
import matplotlib.mlab as mlab

np.random.seed(0)
#import data from a known dataset
dt = pd.read_csv("Database.csv")
#indexing input data
x = dt[["Carbohidratos (g)","Lípidos (g)","Proteína (g)","Sodio (mg)"]]
y = dt["Calorias (kcal)"]
# #make sub-samples of traning and test data using a proportion of 1/3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3)
print('Elements in training sample: %d'%len(x_train))
print('Elements in test     sample: %d'%len(x_test))
#creating a model
print('\nBuilding linear regression model\n')
model = LinearRegression()
#training model
print('\ntraining model ...\n')
model.fit(x_train,y_train)
#print the coefficients
print('Coefficients: \n', model.coef_)
#print intercept
print('intercept: \n',model.intercept_)
#print model
print('\nModel:\nY(x) = %fx1 %fx2 %fx3 %fx4 + %f$'%(model.coef_[0],model.coef_[1],model.coef_[2],model.coef_[3],model.intercept_))
#getting predictions from the model, note that for this we use the test sample,
#not the training sample
model_pred = model.predict(x_test)
npred = 5;

print('\nshowing first %d predictions:\n'%npred)
print(model_pred[:npred])
#print mean squared error
print('\nMean squared error: %.2f'%mean_squared_error(y_test, model_pred))
#print the determination coefficient
print('Coefficient of determination: %.2f'%r2_score(y_test,model_pred))
#let's plot the data, remember that model.predict( ... is just to apply the model
#to any set of input data, it does not mean that we are training the dat again
y_model = model.predict(x_train) # this is the model, and it will be unique

####
####
####
#### Optimization
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)
train_errors, val_errors = [], []
#creating a model
print('\nBuilding linear regression model\n')
model = LinearRegression()
####optimizing in test size
print("optimizing test sample size ...")
#setting a variable to store min value for error
minErr = 1e-6

for m in range(1, len(x_train)):
        if (m%20==0):
         print("training set size: %i"%m)
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_test)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_test, y_val_predict))
        if( m/len(y) > 0.60 ):
            if mean_squared_error(y_test, y_val_predict) < minErr:
                minErr = mean_squared_error(y_test, y_val_predict)
t_size = len(y)-m
        
minErr = np.sqrt(minErr)
print("Min error: %f \n "%minErr)
print('\n Test size from optimization: %.2f'%t_size)

plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="training set")
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="validation set")
plt.legend(loc='best', frameon=False)
plt.ylabel('RMSE') #Raíz del error cuadrático medio
plt.xlabel('training set size')
plt.show()

 #Validation 1: The sum of the residuals is 0
calories = y
predictions = model.predict(x)
n = npred
print("Calorias:")
print(calories)
residuals = calories.values.tolist() - predictions
residualsSum = sum(residuals)
print('\nResiduals:')
print(residuals[:n])
print('\nSum', residualsSum)
val1limit = 1e-3
if abs(residualsSum) < val1limit:
    print('\033[32mValidation #1 is OK\033[0m')
else:
     print('\033[31m####### Validations #1 is not OK: %f > %f ###########\033[0m'%(residualsSum,val1limit))

# ####
# ####
# ####



 #Validation #2: Residulas distribute normaly
num_bins = 15
nsigma = 3
mean_r,std_r=norm.fit(residuals)
 #parameters of residuals dist
print('\nValidation#2\nmean, std: %f, %f'%(mean_r,std_r))
 #plot the residuals hist
n, bins, patches = plt.hist(residuals, num_bins, density='True',facecolor='blue', alpha=0.5)
plt.xlabel('Residuos')
plt.ylabel("N Coincidencias")
plt.title("Prueba de Normalidad")
plt.xlim((mean_r - nsigma*std_r,mean_r + nsigma*std_r))
#fit the residuals hist
xmin, xmax = plt.xlim()
X = np.linspace(xmin, xmax, 100)
y = norm.pdf(X, mean_r, std_r)
plt.plot(X, y,'k',linewidth=2)
plt.grid(True)
plt.show()

# ####
# ####
# ####

#Validation #3: Prueba de varianza homogenea
std_cal = st.stdev(calories) #calcula desviación estándar
residualsV = (calories.values.tolist() - predictions) / std_cal
plt.scatter(predictions,residualsV)
plt.xlabel("Predicción Calorias")
plt.ylabel("Residuos/DEstandar")
plt.title("Varianza Homogénea")
plt.grid(True)
plt.show()

#Validation #4: Prueba Independencia a los Residuos
fig, ax = plt.subplots()
ax.plot(residuals)
plt.xlabel("Predicción Calorias")
plt.ylabel("Residuos")
plt.title("Independencia a los Residuos")
plt.grid(True)
plt.show()

dt.hist()
plt.show()

