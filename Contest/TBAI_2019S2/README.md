df_train.shape = (60000, 235)   
df_test.shape = (10000, 234)

dtypes = {'object': 1, 'int64': 197, 'float64': 37}

NA data (%):    
> parking_area             94.8%    
> parking_price            76.8%    
> txn_floor                26.5%    
> village_income_median     1.9%    

Catrgorical features (2 < np.unique < 20): 50    
> max. number of categories for one feature: 17
> min. number of categories for one feature: 3    
> avg. number of categories for one feature: 9.32
