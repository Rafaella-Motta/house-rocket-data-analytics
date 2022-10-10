import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px
from datetime import datetime

st.set_page_config(layout='wide')

# Funções
st.cache(allow_output_mutation= True)
def get_data(path):
    data = pd.read_csv(path)

    return data

st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def set_feature(data):
    data['price_m2'] = data['price'] / (data['sqft_lot'] * 0.093)
    return data

def overview_data(data):
    # Data Overview

    f_attributes = st.sidebar.multiselect('Enter Columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter Zipcode',
                                       data['zipcode'].sort_values().unique())

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]
    else:
        data = data.copy()

    st.title('Data')
    st.write(data)
    st.write(f_attributes)
    st.write(f_zipcode)

    # Average Metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/M2']

    st.title('Data Overview')

    c1, c2 = st.columns((1.5, 2))

    c1.header('Overview by Zipcode')
    c1.dataframe(df, height=600)

    # Statistic Descriptive
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))

    desvio = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df5 = pd.concat([max_, min_, media, mediana, desvio], axis=1).reset_index()
    df5.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']
    c2.header('Average Values per Attribute')
    c2.dataframe(df5, height=600)

    return None

def portfolio_density(data, geofile):
    c3, c4 = st.columns((1, 1))
    c3.header('Portfolio Density')
    df = data

    # Base Map - Folium

    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()],
                             width=600,
                             height=500,
                             default_zoom_start=15)
    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Price US$ {0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built'])).add_to(marker_cluster)

    with c3:
        folium_static(density_map)

    # Region Price Map
    c4.header('Price Density')
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()

    df.columns = ['ZIP', 'PRICE']

    # df= df.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                  width=600,
                                  height=500,
                                  default_zom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c4:
        folium_static(region_price_map)

    return None

def commercial_distribution(data):
    # Distribução dos imóveis por categorias comerciais
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # Filters
    min_yr_built = int(data['yr_built'].min())
    max_yr_built = int(data['yr_built'].max())
    st.sidebar.subheader('Select Max Year Built')
    f_yr_built = st.sidebar.slider('Year Built',
                                   min_yr_built,
                                   max_yr_built,
                                   max_yr_built)

    st.header('Average Price per Year Built')
    df = data.loc[data['yr_built'] < f_yr_built]

    # Average price per year
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average price per day
    st.header('Average Price per Day')
    st.sidebar.subheader('Select Max Date')

    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, max_date)
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')
    # Filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    median_price = int(data['price'].median())

    f_price = st.sidebar.slider('Price', min_price, max_price, median_price)

    df = data.loc[data['price'] < f_price]

    # data plot

    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):
    # Distribuição dos imóveis por categorias físicas

    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', data['bedrooms'].sort_values().unique())
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', data['bathrooms'].sort_values().unique())
    f_floors = st.sidebar.selectbox('Max number of floors', data['floors'].sort_values().unique())
    f_waterview = st.sidebar.checkbox('Only Houses with Waterview')

    c1, c2 = st.columns(2)
    # Houses per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=33)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses per bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # Houses per floors
    c1.header('Houses per floor')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses per waterview
    data['is_waterfront'] = data['waterfront'].apply(lambda x: 'Yes' if (x == 1) else 'No')
    if f_waterview:
        df = data[data['is_waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='is_waterfront', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    return None

if __name__ == '__main__':
    # ETL
    # data extration
    data = get_data('kc_house_data.csv')

    geofile = get_geofile('https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson')

    # transformation
    data = set_feature(data)

    portfolio_density(data, geofile)

    commercial_distribution(data)

    attributes_distribution(data)

    #loading

