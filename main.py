import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def get_data(filename):
    trips = pq.read_table(filename) 
    return trips

with header:
    st.title('Welcome to my Data Analytics project!')
    st.text("For this project, I used the New York City taxi transactions that took place \nin January 2023")
    
with dataset:
    #st.header('NYC taxi dataset')
    st.text('The dataset comes from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page')
    
    trips = get_data('input/trips.parquet')
    trips = trips.to_pandas()
        
    if st.checkbox('Show data overview'):
        st.subheader('NYC taxis dataset overview')
        st.write(trips.head())
    
    st.subheader('Number of pickups by location ID')
    pu_location_dist = pd.DataFrame(trips['PULocationID'].value_counts()).head(50)
    st.bar_chart(pu_location_dist)
    
    
with features:
    st.subheader("We'll apply the Random Forest Regressor to the NYC taxis dataset")
    st.markdown('##### And check the performance using:')
    st.markdown('* **Mean Absolute Error**')
    st.markdown('* **Mean Squared Error**')
    st.markdown('* **R2 Score**')

with model_training:
    st.header("Let's train the model!")
    st.text('Please choose the hyperparameters of the model and see how the performance changes:')
    
    sel_col, disp_col = st.columns(2)
    
    max_depth = sel_col.slider('What should be the max depth of the model?', min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox('How many trees would you like to use?', options=[10, 50, 100, 200, 300, 'No limit'], index=0)
    
    sel_col.text('List of available features:')
    sel_col.write(trips.columns)
    
    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'PULocationID')
    
    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    X = trips[[input_feature]]
    y = trips[['trip_distance']]
    
    regr.fit(X, y)
    prediction = regr.predict(y)
    
    #disp_col.subheader('Mean Absolute Error of the model is:')
    disp_col.markdown('##### Mean Absolute Error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))
    
    #disp_col.subheader('Mean Squared Error of the model is:')
    disp_col.markdown('##### Mean Squared Error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))
                   
    #disp_col.subheader('R2 score of the model is:')
    disp_col.markdown('##### R2 score of the model is:')
    disp_col.write(r2_score(y, prediction))
    
    
    

    
 
    
    
    
    
    
