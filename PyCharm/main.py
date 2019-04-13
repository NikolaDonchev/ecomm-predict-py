import pandas as pd
import turicreate as tc
import mysql.connector as db
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Connecting to the MySQL database
mydb = db.connect(
  host = "localhost",
  user = "root",
  passwd = "",
  database = "ecomm_ml"
)

# Preparing initial data
lineItems = mydb.cursor()
lineItems.execute("SELECT `customer_id`, `product_id`, `quantity` FROM line_items")
lineItems = lineItems.fetchall()
customers = mydb.cursor()
customers.execute("SELECT `id` from `customers`")
customers = customers.fetchall()

# Creating a pandas DataFrame for the data
lineItems = pd.DataFrame(lineItems, columns=['customer_id', 'product_id', 'quantity'])

# Get only the ID of the customer from the SQL query
customers_list = []
for i in customers:
    customers_list.append(i[0])

# Sum all purchases of the same item from the same customer
lineItems = lineItems.groupby(['customer_id', 'product_id'], as_index=False)['quantity'].sum()

# Functions that will be used
def minmax_normalisation(data):
    '''
    Min-Max Normalisation
    https://www.codecademy.com/articles/normalization
    :param data:
    :return matrix_normalised:
    '''
    # Before we use the formula it would be easier to pivot the table to a matrix
    matrix = pd.pivot_table(data, values='quantity', index='customer_id', columns='product_id')
    matrix_normalised = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    # Reset the index
    matrix_normalised = matrix_normalised.reset_index()
    # Create a new index column in the data for the minmax quantity
    matrix_normalised.index.names = ['minmax_quantity']
    # Use pandas melt to change the matrix back to a list
    matrix_normalised = pd.melt(matrix_normalised, id_vars=['customer_id'], value_name='minmax_quantity').dropna()
    return matrix_normalised

def ml_split(items):
    '''
    Split the data for ML model 80:20 ratio
    and create SFrame
    :param items:
    :return train_data, test_data:
    '''
    train, test = train_test_split(items, test_size=.2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data

def generate_recommendations(data, customers, alg, target=None):
    train, test = ml_split(data)
    print(train)
    if alg == "popularity":
        model = tc.popularity_recommender.create(train, user_id="customer_id", item_id="product_id", target=target)

    elif alg == "similarity":
        model = tc.item_similarity_recommender.create(train, user_id="customer_id", item_id="product_id", target=target, similarity_type='cosine')

    recommendations = model.recommend(users=customers, k=10)
    return recommendations

lineItemsMinMax = minmax_normalisation(lineItems)

# train, test = train_test_split(lineItems, test_size = .2)
# train_data = tc.SFrame(train)
# test_data = tc.SFrame(test)

popularity_recommendations_normal = generate_recommendations(lineItems, customers_list, "popularity", "quantity")
popularity_recommendations_minmax = generate_recommendations(lineItemsMinMax, customers_list, "popularity", "minmax_quantity")
popularity_recommendations_noQuantity = generate_recommendations(lineItems, customers_list, "popularity")

popularity_recommendations_normal.print_rows(80)