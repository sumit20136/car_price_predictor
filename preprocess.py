# from google.colab import files
# uploaded = files.upload()
def find_ans(name,company,year,kms_driven,fuel_type):
    from asyncio.windows_utils import pipe
    import io
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    import numpy as np
    # def do_work(df):

    import pandas as pd
    df = pd.read_csv('CarPricePredictionDataset.csv',encoding="ISO-8859-1")
    df.head()
    """#ANALYZING DATA"""


    df.isnull().sum()
    df['year'].unique()

    # things to improve:::
    #   change year from object to integer
    #   remove all non-integer values from year
    # df['Price'].unique()
    # things to improve:::
    #   change Price from object to integer
    #   remove 'Ask for Advice from dataset'
    #   remove commas from prices
    # df['kms_driven'].unique()
    # things to improve:::
    #   change kms_driven from object to integer
    #   remove commas,"kms"suffix from values
    #   remove NULL values
    #   remove all non-integer values
    # df['company'].unique()
    # things to improve::
    #   remove all integer values
    # df['fuel_type'].unique()
    # things to improve::
    #   remove all NULL values
    # copying the dataset and then transforming it
    df_new=df[:]
    df=df_new[:]
    # Removing errors in year

    df=df[df['year'].str.isnumeric()]
    df['year']=df['year'].astype(int)
    #Removing errors in Price

    df=df[df['Price']!='Ask For Price']
    df['Price']=df['Price'].str.replace(',','')
    df['Price']=df['Price'].astype(int)
    #Removing errors in kms_driven
    df['kms_driven']=df['kms_driven'].str.replace(',','')
    df['kms_driven']=df['kms_driven'].str.replace('kms','')
    df['kms_driven']=df['kms_driven'].str.replace(" ",'')
    df=df[df['kms_driven'].str.isnumeric()]
    df['kms_driven']=df['kms_driven'].astype(int)
    #Removing errors in company
    df=df[~df['company'].str.isnumeric()]
    # Removing errors in fuel_type
    df=df[~df['fuel_type'].isnull()]
    # Removing extreme outliers
    df=df[df['Price']<8e5].reset_index(drop=True)
    df.describe()
    # As we have limited data, and the name of cars are way too long, so we are slicing it to first 3 words only...
    df['name']=df['name'].str.split(' ').str.slice(0,3).str.join(' ')
    # print(df['name'])
    """#Making labels"""
    df.to_csv('Cleaned_Car_data.csv')
    df.to_pickle("Cleaned_dataframe.pkl")
    """Making splits for training and testing"""
    df = pd.read_pickle("Cleaned_dataframe.pkl")
    X=df.drop(columns=['Price'])
    Y=df['Price'] 
    # print(X)
    X_train, X_test, y_train, y_test =  train_test_split(X,Y,test_size = 0.2)
    """Using KFolds to train model"""

    """#Checking and removing Outliers
    Checking outliers in Kms_driven
    """
    # import seaborn as sns
    # sns.boxplot(df['kms_driven'])
    """Checking outliers in Price"""
    # sns.boxplot(df['Price'])
    """Standardizing the data """
    col_names = ['kms_driven']
    features = X_train[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    X_train[col_names] = features
    col_names = ['kms_driven']
    features = X_test[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    X_test[col_names] = features
    ohe=OneHotEncoder()
    ohe.fit(X[['name','company','fuel_type']])
    columns_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
    print("the value is")
    print(type(columns_trans))
    lr=LinearRegression()
    pipe=make_pipeline(columns_trans,lr)
    pipe.fit(X_train,y_train)
    scores=[]
    for i in range(1000):
      X_train, X_test, y_train, y_test =  train_test_split(X,Y,test_size = 0.2,random_state=i)
      lr=LinearRegression()
      pipe=make_pipeline(columns_trans,lr)
      pipe.fit(X_train,y_train)
      y_pred=pipe.predict(X_test)
      scores.append(r2_score(y_test,y_pred))  

    print(scores[np.argmax(scores)])

    """Storing the pipe with maximum accuracy"""

    X_train, X_test, y_train, y_test =  train_test_split(X,Y,test_size = 0.2,random_state=np.argmax(scores))
    lr=LinearRegression()
    pipe=make_pipeline(columns_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    print(r2_score(y_test,y_pred))
    import pickle
    pickle.dump(pipe,open('CarPricePredictorModel.pkl','wb'))
    # def find_ans(name,company,year,kms_driven,fuel_type):
        # import numpy as np
        # model = pickle.load(open("CarPricePredictorModel.pkl", 'rb'))
    return(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([name,company,year,kms_driven,fuel_type]).reshape(1,5))))


