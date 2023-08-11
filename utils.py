
import pandas as pd
from sklearn.impute import KNNImputer

def get_string_cols(df):
    return df.select_dtypes(include=['object']).columns.tolist()

def factorize(df, exclude = []):
    for col in get_string_cols(df):

        if col not in exclude:

            df[col] = pd.factorize(df[col])[0]
    return df



def impute(data: pd.DataFrame, method : str = 'knn', target_var ='Transported', exclude = []) -> pd.DataFrame:
    '''
    Impute missing values in a dataframe using the specified method
    args:
        data: dataframe to impute
        method: method to use for imputation
        target_var: target variable to drop and re-add after imputation if it exists in the dataframe
    returns:
        data: dataframe with imputed values
    
    '''
    
    ignore_target_var = target_var in data.columns
    
    if ignore_target_var:
        y = data[target_var]

        data = data.drop(target_var, axis=1)
    if len(exclude) > 0:

        exc= data[exclude]
        data = data.drop(exclude, axis = 1)
        

    if method == 'mean':
        data = data.fillna(data.mean())
    elif method == 'median':
        data =data.fillna(data.median())
    elif method == 'mode':
        data = data.fillna(data.mode())

    elif method == 'knn':
        #impute the data using sklearn.impute.KNNImputer
        imputer = KNNImputer(n_neighbors=10)
        data = pd.DataFrame(imputer.fit_transform(data), columns = data.columns)
    else:
        raise ValueError('Method not recoginzed')

    if ignore_target_var:
        data[target_var] = y

    if len(exclude) > 0:
        data = pd.concat([data, exc], axis = 1)

    return data