import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing 
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from sklearn.feature_selection import SelectKBest, f_regression


## acquire data

def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the sql database.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


def get_telco_data():
    '''
    This function reads in the telco churn data from the sql database
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    SELECT *
    FROM payment_types
    JOIN customers ON payment_types.payment_type_id = customers.payment_type_id
    JOIN internet_service_types ON internet_service_types.internet_service_type_id = customers.internet_service_type_id
    JOIN contract_types ON contract_types.contract_type_id = customers.contract_type_id;
    '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))



def prep_telco(df):
    '''
    This function takes in the telco dataframe and specifies it to two year customers.
    It also replaces whitespaces with nulls, drops nulls, and changes total charges to
    a float. In the end it only returns specified columns.
    '''
    # isolate the customers on two year plans
    df = df[df['contract_type']=='Two year']

    # assign my dataframe to the columns I want
    column = ['customer_id', 'monthly_charges', 'tenure', 'total_charges']
    df = df[column]

    # replace my empty whitespaces with nan values
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # drop my nulls
    df = df.dropna()

    # change my total charges column to a float
    df.total_charges = df.total_charges.astype('float')

    # replace 0 tenure months with 1
    df.tenure = df.tenure.replace(0, 1)

    # reset index
    df.reset_index(inplace=True)

    return df


def telco_split(df):
    '''
    This function takes in a dataframe and splits it into train, test, and validate dataframes for my model
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('train--->', train.shape)
    print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, validate, test



def split_data(df):
    '''
    This function takes in a dataframe and splits it into train, test, and validate dataframes for my model
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('train--->', train.shape)
    print('validate--->', validate.shape)
    print('test--->', test.shape)
    return train, validate, test


## *********** Zillow data ****************

def get_zillow():
    '''
    This function reads in the zillow data from the sql database
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    SELECT bathroomcnt, 
			calculatedfinishedsquarefeet,
			taxvaluedollarcnt,
			yearbuilt,
			taxamount,
			fips
    FROM properties_2017
    WHERE propertylandusetypeid LIKE 261;
    '''
    return pd.read_sql(sql_query, get_connection('zillow'))



def prep_zillow(df):
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df




## MinMax Scaler function

def minmax_scaler(train, validate, test):

    # create my variable to columns
    cols = ['monthly_charges', 'total_charges', 'tenure']
    
    # assign my scaler
    scaler = MinMaxScaler()
    
    for col in cols:
    
        # apply fit_transform to train dataset
        train[col] = scaler.fit_transform(train[[col]])
        
        # apply transform to validate and test set
        validate[col] = scaler.transform(validate[[col]])
        test[col] = scaler.transform(test[[col]])
    
    return train, validate, test



# The better function for minmax scaler


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    return X_train_scaled, X_validate_scaled, X_test_scaled




def select_kbest(X_train_scaled, y_train, no_features):
    
    # using kbest
    f_selector = SelectKBest(score_func=f_regression, k=no_features)
    
    # fit
    f_selector.fit(X_train_scaled, y_train)

    # display the two most important features
    mask = f_selector.get_support()
    
    return X_train_scaled.columns[mask]



def rfe(X_train_scaled, y_train, no_features):
    # now using recursive feature elimination
    lm = LinearRegression()
    rfe = RFE(estimator=lm, n_features_to_select=no_features)
    rfe.fit(X_train_scaled, y_train)

    # returning the top chosen features
    return X_train_scaled.columns[rfe.support_]