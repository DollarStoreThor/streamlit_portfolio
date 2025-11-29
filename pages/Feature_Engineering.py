import streamlit as st

st.markdown("""
## Notebook set up


**Tasks**: Apply some basic feature engineering techniques to the `housing_df` dataframe to improve the dataset. At the end of the notebook, my engineered dataset, a validation, and the original dataset will be used to train a linear regression model to predict `MedHouseVal`. My goal is to achieve better model performance on the engineered and validation datasets compared to the original, only via feature engineering.

Before applying transformations, I explore the dataset to understand what techniques would be most beneficial.

### Import libraries
""")

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Custom imports
import pygeohash as pgh
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, PowerTransformer, QuantileTransformer

# Set random seed for reproducibility
np.random.seed(315)


st.markdown("""
```python
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Custom imports
import pygeohash as pgh
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, PowerTransformer, QuantileTransformer

# Set random seed for reproducibility
np.random.seed(315)
""")

st.markdown("""### Load dataset""")

# Load California housing dataset
original_housing_df = pd.read_csv('https://gperdrizet.github.io/FSA_devops/assets/data/unit2/california_housing.csv')
housing_df = original_housing_df.copy()

st.markdown("""
```python
# Load California housing dataset
original_housing_df = pd.read_csv('https://gperdrizet.github.io/FSA_devops/assets/data/unit2/california_housing.csv')
housing_df = original_housing_df.copy()
""")

st.markdown("""
## Task 1: Explore the Dataset

Before deciding what feature engineering techniques to apply, I explore the dataset to understand its characteristics.

**Things I investigate**:
- Display basic information about the dataset (`.info()`, `.describe()`)
- Check for missing values
- Examine feature distributions (histograms, box plots)
- Look at feature scales and ranges

I use this exploration to inform my feature engineering decisions in the following code blocks.""")


st.markdown("""
```python
housing_df.sample(20)
""")
st.dataframe(housing_df.sample(20))


st.markdown("""
### Sample view
- MedInc - 1.8715 to 11.1567 - Median Income
- HouseAge - 2.0 to 52.0 - House Age
- AveRooms - 2.942116 to 8.813167	- Average num Rooms ---> Clip greater than 25 to 25
- AveBedrms - 0.917840	to 1.187970 - Average num Bedrooms ----> Clip greater than 5 to 5
- Population -	614.0 to 2212.0 - City Population ----> Clip greater than 10,000 to 10,000
- AveOccup - 2.445110 to 4.776243 - Average num People in House  ----> Clip greater than 10 to 10
- Latitude - 33.55 to 38.68           Bin lat long to geohash ()
- Longitude	- -122.43 to -117.34
- MedHouseVal - 0.71400 to 5.00001 - Median House Value (Target)
""")


st.markdown("""
```python
housing_df.describe()
""")
st.dataframe(housing_df.describe())


st.markdown("""
```python
to_chart = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]

# Determine relationship between extreme MEDHOUSVAL and other features
extreme_highend_housing_df = housing_df[(housing_df['MedHouseVal'] >= 5)]
extreme_highend_housing_df

extreme_lowend_housing_df = housing_df[(housing_df['MedHouseVal'] <= 1)]


# Explore features with histograms and PDFs on the same chart
for feature in to_chart:
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot histogram with density normalization
    ax.hist(
        extreme_highend_housing_df[feature],
        bins=300,
        density=True,
        alpha=0.7,
        label='High-end (MedHouseVal >= 5)',
        color='green'
    )
    ax.hist(
        housing_df[feature],
        bins=300,
        density=True,
        alpha=0.4,
        label='All Data',
        color='blue'
    )
    ax.hist(
        extreme_lowend_housing_df[feature],
        bins=300,
        density=True,
        alpha=0.4,
        label='Low-end (MedHouseVal <= 1)',
        color='red'
    )
    
    # Plot PDF for high-end subset
    extreme_highend_housing_df[feature].plot(
        kind='density',
        ax=ax,
        linewidth=2,
        label='High-end PDF',
        color='green',
        alpha=0.5
    )
    # Plot PDF for full data
    housing_df[feature].plot(
        kind='density',
        ax=ax,
        linewidth=2,
        label='All Data PDF',
        color='blue',
        alpha=0.5
    )
    # Plot PDF for low-end subset
    extreme_lowend_housing_df[feature].plot(
        kind='density',
        ax=ax,
        linewidth=2,
        label='Low-end PDF',
        color='red',
        alpha=0.5
    )
    
    ax.set_title(f"Binned Count and PDF of {feature} -- Pre-modifications")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.suptitle('Extreme High, Low, and All Data Values for MedHouseVal', y=1.02)
    plt.show()
    """)

to_chart = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]

# Determin relationship between extreme MEDHOUSVAL and other features
extreme_highend_housing_df = housing_df[(housing_df['MedHouseVal'] >= 5)]
#extreme_highend_housing_df

extreme_lowend_housing_df = housing_df[(housing_df['MedHouseVal'] <= 1)]


# Explore features with histograms and PDFs on the same chart
for feature in to_chart:
    st.image(f"pages/FE_res/basic/extreme_high_low_medhouseval_{feature}.png", caption=f'Feature', use_container_width=True)


st.markdown("""
```python
# See if I can get the shape of Callifornia
coords = housing_df[['Latitude', 'Longitude']].values

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 9))
scatter = plt.scatter(coords[:, 1], coords[:, 0], s=1000/housing_df['Population'], cmap='rainbow', alpha=0.6, c=housing_df['MedHouseVal'])
plt.legend(*scatter.legend_elements(), title="Scaled MedHouseVal")
plt.colorbar(label='MedHouseVal')
plt.title('California Housing Locations with Population Size')
plt.savefig('california_housing_locations.png')
""")

st.image('pages/FE_res/california_housing_locations.png', caption='California Housing Locations with Population Size', use_container_width=True)


st.markdown("""
## Task 2: Apply My First Feature Engineering Technique

Based on my exploration, apply my first feature engineering technique.

**Example approaches**:
- Transform skewed features using log, sqrt, power, or quantile transformations
- Create bins/categories from continuous variables
- Create interaction features (e.g., rooms per household = total rooms / households)
""")

st.markdown("""
```python
'''Perform the following operations:
- MedInc - 1.8715 to 11.1567 - Median Income
- HouseAge - 2.0 to 52.0 - House Age
- AveRooms - 2.942116 to 8.813167	- Average num Rooms ---> Clip greater than 25 to 25
- AveBedrms - 0.917840	to 1.187970 - Average num Bedrooms ----> Clip greater than 5 to 5
- Population -	614.0 to 2212.0 - City Population ----> Clip greater than 10,000 to 10,000
- AveOccup - 2.445110 to 4.776243 - Average num People in House  ----> Clip greater than 100 to 100
'''

housing_df['AveRooms'] = housing_df['AveRooms'].clip(upper=25)
housing_df['AveBedrms'] = housing_df['AveBedrms'].clip(upper=5)
housing_df['Population'] = housing_df['Population'].clip(lower= 10, upper=10000)       # Current Best is lower= 10, upper=10000
housing_df['AveOccup'] = housing_df['AveOccup'].clip(upper=10)                         # Current Best is  upper=10
housing_df['MedInc'] = housing_df['MedInc'].clip(lower= 1.2, upper=10.5)               # Current Best is lower= 1.2, upper=10.5

housing_df.describe()
""")

housing_df['AveRooms'] = housing_df['AveRooms'].clip(upper=25)
housing_df['AveBedrms'] = housing_df['AveBedrms'].clip(upper=5)
housing_df['Population'] = housing_df['Population'].clip(lower= 10, upper=10000)       # Current Best is lower= 10, upper=10000
housing_df['AveOccup'] = housing_df['AveOccup'].clip(upper=10)                         # Current Best is  upper=10
housing_df['MedInc'] = housing_df['MedInc'].clip(lower= 1.2, upper=10.5)               # Current Best is lower= 1.2, upper=10.5

st.dataframe(housing_df.describe())








