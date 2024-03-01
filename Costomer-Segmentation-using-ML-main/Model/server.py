import pickle
import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd

st.set_page_config(layout="wide")

def load_lottifiel(filepath:str):
    with open(filepath,'r') as f:return json.load(f)
st_lottie( load_lottifiel("Media/Animation - 1708670185993.json"),height=250)

Cluster = {
      0 : """1. Are a parent
2. At the max have 4 members in the family and at least 2
3. Single parents are a subset of this group
4. Most have a teenager at home
5. Relatively older""",

1 : """1. Are not a parent
2. At the max are only 2 members in the family
3. slight majority of couples over single people 
4. Span all ages
5.  A high income group""",

2 : """1. The majority of these people are parents
2.  At the max are 3 members in the family
3. They majorly have one kid (and not teenagers, typically) 
4. Relatively younger""",

3 : """1. They are a parent
2. At the max are 5 members in the family and at least 2 
3. The majority of them have a teenager at home
4. Relatively older
5. A lower-income group"""
}

model = pickle.load(open("Model/agglome.obj",'rb'))
pca = pickle.load(open("Pipeline/pca.obj",'rb'))
scaler = pickle.load(open("Pipeline/scaler.obj",'rb'))

education_label =  pickle.load(open("Pipeline/Education-label.obj",'rb'))
living_label =  pickle.load(open("Pipeline/Living_With-label.obj",'rb'))

columns = ['Education', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'Wines',
       'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'Age', 'Spent', 'Living_With',
       'Children', 'Family_Size']

data = []

# Education Qualification of customer -- Education {'Graduate', 'Postgraduate', 'Undergraduate'}
Education = st.selectbox(
    'Education Qualification of customer',
    ('Graduate', 'Postgraduate', 'Undergraduate'))
data.append(Education)

# Customer's yearly household income -- Income
Income = st.number_input("Customer's yearly household income",step=1,min_value=0)
data.append(Income)

# Number of children in customer's household -- Kidhome
Kidhome = st.number_input("Number of children in customer's household",step=1,min_value=0)
data.append(Kidhome)

# Number of teenagers in customer's household -- Teenhome
Teenhome = st.number_input("Number of teenagers in customer's household",step=1,min_value=0)
data.append(Teenhome)

# Number of days since customer's last purchase -- Recency
Recency = st.number_input("Number of days since customer's last purchase",step=1,min_value=0)
data.append(Recency)


# Amount spent on wine -- Wines
Wines = st.number_input("Amount spent on wine",step=1,min_value=0)
data.append(Wines)

# Amount spent on Fruit -- Fruits
Fruits = st.number_input("Amount spent on Fruit",step=1,min_value=0)
data.append(Fruits)

# Amount spent on Meat -- Meat
Meat = st.number_input("Amount spent on Meat",step=1,min_value=0)
data.append(Meat)

# Amount spent on Fish -- Fish
Fish = st.number_input("Amount spent on Fish",step=1,min_value=0)
data.append(Fish)

# Amount spent on Sweet -- Sweets
Sweets = st.number_input("Amount spent on Sweet",step=1,min_value=0)
data.append(Sweets)

# Amount spent on Gold -- Gold
Gold = st.number_input("Amount spent on Gold",step=1,min_value=0)
data.append(Gold)

# Number of purchases made using some kind of deal or promotion -- NumDealsPurchases
NumDealsPurchases = st.number_input("Number of purchases made using some kind of deal or promotion",step=1,min_value=0)
data.append(NumDealsPurchases)

# Number of purchases made through a website or online platform -- NumWebPurchases
NumWebPurchases = st.number_input("Number of purchases made through a website or online platform",step=1,min_value=0)
data.append(NumWebPurchases)

# Number of purchases made with the application of deals or discounts -- NumCatalogPurchases
NumCatalogPurchases = st.number_input("Number of purchases made with the application of deals or discounts",step=1,min_value=0)
data.append(NumCatalogPurchases)

# Number of purchases made directly from physical stores. -- NumStorePurchases
NumStorePurchases = st.number_input("Number of purchases made directly from physical stores",step=1,min_value=0)
data.append(NumStorePurchases)

# Number of visits or sessions to a website within a specific month -- NumWebVisitsMonth
NumWebVisitsMonth = st.number_input("Number of visits or sessions to a website within a specific month",step=1,min_value=0)
data.append(NumWebVisitsMonth)

# Customer's age -- Age
Age = st.number_input("Customer's Age",step=1,min_value=0)
data.append(Age)

### Wines + Fruits + Meat + Fish + Sweets + Gold -- Spent
Spent = Wines + Fruits + Meat + Fish + Sweets + Gold
data.append(Spent)

# Living with -- Living_With {'Alone', 'Partner'}
Living_With = st.selectbox(
    'Living with',
    ('Alone', 'Partner'))
data.append(Living_With)

### Kidhome + Teenhome -- Children
Children = Kidhome + Teenhome
data.append(Children)

### ()"Alone": 1, "Partner":2)+ Children -- Family_Size
if(Living_With == "Alone"):Family_Size = 1+Children
else:Family_Size = 2+Children
data.append(Family_Size)





if st.button('Submit'):
       data = pd.DataFrame([data],columns= columns)
       data['Education'] = education_label.transform(data['Education'])
       data['Living_With'] = living_label.transform(data['Living_With'])
       data = pd.DataFrame(scaler.transform(data),columns= columns)
       data = pd.DataFrame(pca.transform(data), columns=(["col1","col2", "col3"]))
       predict = model.predict(data)[0]
       
       st.title(f"The Customer belongs to the group {predict+1}")
       st.write(Cluster[predict])
