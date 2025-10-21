import pandas as pd
import numpy as np
from scipy.stats import f, t


df = pd.read_csv('hpriice_ohe.csv')







df.drop(columns='Neighborhood_Rural')

cols= ['Neighborhood_Suburb','Neighborhood_Urban','SquareFeet','Bedrooms','Bathrooms','YearBuilt','intercept']

subset_X= df[cols]

Y_Matrix= df['Price'].to_numpy(dtype=float)


XMatrix=subset_X.to_numpy(dtype=float)





XMatrix_Transposed= XMatrix.T


B_Hat= (np.linalg.inv(XMatrix_Transposed @ XMatrix)) @ (XMatrix_Transposed @ Y_Matrix)




print(B_Hat)


#need to find y hat( predicted y values) to do this u do the x column times by the beta matrix we have made. this gives y hat.
#we need y hat to find the reiduals. r= y-yhat

yhat= XMatrix @ B_Hat

df['yhat']=yhat

# print(df['yhat'])

df['Residuals']= Y_Matrix - yhat

# print(df['Residuals'])


sumsquared= np.sum(df['Residuals']**2)

print(sumsquared)

Cjj= np.linalg.inv(XMatrix_Transposed@XMatrix)

# ------------------------- Variables for dataset

n= df.shape[0]
parameters_k= df.shape[1]-3 # y value, yhat, residuals. these are not parameters but we have added them to our data set.
print(f'number of rows= {n}')
print(n,parameters_k)
#-----------------------------------

standard_error= np.sqrt(sumsquared/(n-parameters_k))

print(standard_error)


print(Cjj)



standard_error_of_coefficient_b1= np.sqrt((standard_error**2)*5.99682756e-05) #diagonal because of Cjj, jj is diagonal
standard_error_of_coefficient_b2= np.sqrt((standard_error**2)*1.20214002e-04)
standard_error_of_coefficient_b3= np.sqrt((standard_error**2)*6.03970398e-11)
standard_error_of_coefficient_b4= np.sqrt((standard_error**2)*1.60509063e-05)
standard_error_of_coefficient_b5= np.sqrt((standard_error**2)*3.00512067e-05)
standard_error_of_coefficient_b6= np.sqrt((standard_error**2)*4.65909961e-08)
standard_error_of_coefficient_b0= np.sqrt((standard_error**2)*1.84183279e-01) #y intercept


print(f'B1:{standard_error_of_coefficient_b1}, B2:{standard_error_of_coefficient_b2}, B3:{standard_error_of_coefficient_b3}, B4:{standard_error_of_coefficient_b4}, B5:{standard_error_of_coefficient_b5}, B6:{standard_error_of_coefficient_b6}, B0:{standard_error_of_coefficient_b0}')


#is there a correlation to the input and outputs: SSt n-1 degree of freedom;
#for regression to be significant we would expect most of the variability to be explained by the model so model significance value significantly above 1. SSR/SSE
#SST=SSR+SSE

ybar = sum(df['Price'])/n

print(ybar)


#the residiual sum squared is what we worked out earlier which is SSR or just variable sumsquared of the residuals
sumsquared # this is the sum square of the residuals(errors) so the sum square error

total_sumsquare= np.var(Y_Matrix, ddof=1)*(n-1)

print(total_sumsquare)

sum_square_of_regressor= total_sumsquare-sumsquared

meansquared_regressor= sum_square_of_regressor/parameters_k
meansquared_error= sumsquared/(n-(parameters_k+1))

F0= meansquared_regressor/meansquared_error

print(f'F0={F0}')


p_value= f.sf(F0, parameters_k,(n-(parameters_k+1)))

print(f'pval: {p_value:.50f} When a p-value tends toward 0, it indicates that the observed data are highly unlikely to have occurred by chance, providing strong evidence against the null hypothesis. This suggests that there is a statistically significant effect or relationship in the data.')


t_stat_of_coeficcient_b1=-6.75494104e+02/standard_error_of_coefficient_b1
t_stat_of_coeficcient_b2=-1.55008822e+03/standard_error_of_coefficient_b2
t_stat_of_coeficcient_b3=-9.93400134e+01/standard_error_of_coefficient_b3
t_stat_of_coeficcient_b4=-5.07443485e+03/standard_error_of_coefficient_b4
t_stat_of_coeficcient_b5=-2.83383485e+03/standard_error_of_coefficient_b5
t_stat_of_coeficcient_b6=-1.08868545e+01/standard_error_of_coefficient_b6
t_stat_of_coeficcient_b0=-2.34314072e+04/standard_error_of_coefficient_b0    #y intercept


pval_b1= t.sf(abs(t_stat_of_coeficcient_b1), (n-(parameters_k+1))) * 2
pval_b2=t.sf(abs(t_stat_of_coeficcient_b2), (n-(parameters_k+1))) * 2
pval_b3=t.sf(abs(t_stat_of_coeficcient_b3), (n-(parameters_k+1))) * 2
pval_b4=t.sf(abs(t_stat_of_coeficcient_b4), (n-(parameters_k+1))) * 2
pval_b5=t.sf(abs(t_stat_of_coeficcient_b5), (n-(parameters_k+1))) * 2
pval_b6=t.sf(abs(t_stat_of_coeficcient_b6), (n-(parameters_k+1))) * 2
pval_b0=t.sf(abs(t_stat_of_coeficcient_b0), (n-(parameters_k+1))) * 2


#pval
print(f'pval_b1:{pval_b1}, pval_b2:{pval_b2}, pval_b3:{pval_b3}, pval_b4:{pval_b4}, pval_b5:{pval_b5}, pval_b6:{pval_b6}, pval_b2:{pval_b0}') # any pval over 0.05 is not sig this is how to show if your coefficients are needed