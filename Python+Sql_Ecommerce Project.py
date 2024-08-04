#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import mysql.connector
import os

# List of CSV files and their corresponding table names
csv_files = [
    ('payments.csv', 'payments'),
    ('order_items.csv', 'order_items')# Added payments.csv for specific handling
]

# Connect to the MySQL database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='1234567',
    database='ecommerce'
)
cursor = conn.cursor()

# Folder containing the CSV files
folder_path = 'C:/Users/Alina/Desktop/Projects/ecommace'

def get_sql_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INT'
    elif pd.api.types.is_float_dtype(dtype):
        return 'FLOAT'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'DATETIME'
    else:
        return 'TEXT'

for csv_file, table_name in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Replace NaN with None to handle SQL NULL
    df = df.where(pd.notnull(df), None)
    
    # Debugging: Check for NaN values
    print(f"Processing {csv_file}")
    print(f"NaN values before replacement:\n{df.isnull().sum()}\n")

    # Clean column names
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_') for col in df.columns]

    # Generate the CREATE TABLE statement with appropriate data types
    columns = ', '.join([f'`{col}` {get_sql_type(df[col].dtype)}' for col in df.columns])
    create_table_query = f'CREATE TABLE IF NOT EXISTS `{table_name}` ({columns})'
    cursor.execute(create_table_query)

    # Insert DataFrame data into the MySQL table
    for _, row in df.iterrows():
        # Convert row to tuple and handle NaN/None explicitly
        values = tuple(None if pd.isna(x) else x for x in row)
        sql = f"INSERT INTO `{table_name}` ({', '.join(['`' + col + '`' for col in df.columns])}) VALUES ({', '.join(['%s'] * len(row))})"
        cursor.execute(sql, values)

    # Commit the transaction for the current CSV file
    conn.commit()

# Close the connection
conn.close()


# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector

db=mysql.connector.connect(host = "localhost",
                          username ="root",
                          password = "1234567",
                          database = "ecommerce")
cur = db.cursor()


# # List all unique cities where customers are located.

# In[2]:


query = """ select distinct(customer_city) from customers """

cur.execute(query)

data = cur.fetchall()

df = pd.DataFrame(data)
df.head()


# # Count the number of orders placed in 2017.

# In[3]:


query = """ select count(order_id) from orders where year(order_purchase_timestamp) = 2017  """

cur.execute(query)

data = cur.fetchall()

"Total order placed in 2017 are", data[0][0]


# # Find the total sales per category.

# In[4]:


query = """ select upper(products.product_category) category, 
round(sum(payments.payment_value),2) sales
from products join order_items
on products.product_id = order_items.product_id
join payments
on payments.order_id = order_items.order_id
group by category

"""

cur.execute(query)

data = cur.fetchall()
df = pd.DataFrame(data, columns =["Category","sales"])
df


# # Calculate the percentage of orders that were paid in installments.

# In[5]:


query = """ SELECT (SUM(CASE WHEN payment_installments >= 1 THEN 1 
ELSE 0 END)/ COUNT(*))* 100 FROM payments;
"""

cur.execute(query)

data = cur.fetchall()

"The percentage of orders that were paid in installments is", data[0][0]


# # Count the number of customers from each state.

# In[6]:


query = """ Select customer_state, count(customer_id)
from customers group by customer_state
"""

cur.execute(query)

data = cur.fetchall()

df = pd.DataFrame(data, columns =["state","customer_count"])
df = df.sort_values(by ="customer_count", ascending= False)

plt.figure(figsize =(8,3))
plt.bar(df["state"], df["customer_count"])
plt.xticks(rotation =90)
plt.xlabel("states")
plt.ylabel("customer_count")
plt.title("Count of Customers by Sates")
plt.show()


# # Calculate the number of orders per month in 2018.

# In[7]:


query = """ select monthname(order_purchase_timestamp) months, count(order_id) order_count
from orders where year(order_purchase_timestamp)= 2018
group by months
"""

cur.execute(query)

data = cur.fetchall()
df = pd.DataFrame(data, columns = ["months", "order_count"])
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]

ax = sns.barplot(x = df["months"],y = df["order_count"], data = df, order = month_order )
plt.xticks(rotation = 45)
ax.bar_label(ax.containers[0])
plt.title("Count of Orders by Months is 2018")

plt.show()


# # Find the average number of products per order, grouped by customer city.

# In[8]:


import mysql.connector

# Connect to your database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234567",
    database="ecommerce"
)
cur = conn.cursor()

query = """
WITH count_per_order AS (
    SELECT orders.order_id, orders.customer_id, COUNT(order_items.order_id) AS oc
    FROM orders JOIN order_items
    ON orders.order_id = order_items.order_id
    GROUP BY orders.order_id, orders.customer_id
)

SELECT customers.customer_city, ROUND(AVG(count_per_order.oc), 2) AS average_orders
FROM customers JOIN count_per_order 
ON customers.customer_id = count_per_order.customer_id
GROUP BY customers.customer_city order by average_orders desc
"""
cur.execute(query)

data = cur.fetchall()
df = pd.DataFrame(data, columns = ["Customer City", "Average Product/Orders"])
df.head(10)


# # Calculate the percentage of total revenue contributed by each product category.

# In[9]:


import mysql.connector

query = """
SELECT 
    UPPER(products.product_category) AS category, 
    ROUND(
        (SUM(payments.payment_value) / 
         (SELECT SUM(payment_value) FROM payments)) * 100, 2
    ) AS sales_percentage
FROM products 
JOIN order_items
    ON products.product_id = order_items.product_id
JOIN payments
    ON payments.order_id = order_items.order_id
GROUP BY category 
ORDER BY sales_percentage DESC
"""
cur.execute(query)


df = pd.DataFrame(data, columns=["Category", "percent distribution"])
df.head()


# # Identify the correlation between product price and the number of times a product has been purchased.

# In[10]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
import numpy as np

db=mysql.connector.connect(host = "localhost",
                          username ="root",
                          password = "1234567",
                          database = "ecommerce")
cur = db.cursor()

# Connect to your database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234567",
    database="ecommerce"
)
cur = conn.cursor()

query = """
select products.product_category,
count(order_items.product_id),
round(avg(order_items.price),2)
from products join order_items
on products.product_id = order_items.product_id
group by products.product_category;
"""
cur.execute(query)

data = cur.fetchall()
df = pd.DataFrame(data, columns = ["Category", "order_count","price"])

arr1 = df["order_count"]
arr2 = df["price"]

a = np.corrcoef([arr1,arr2])
print("The coorrelation between price and the number of time a product has been purchased is"),a[0][1]


# # Calculate the total revenue generated by each seller, and rank them by revenue.

# In[11]:


query = """ 
select *, dense_rank() over(order by revenue desc) as rn from
(select order_items.seller_id, sum(payments.payment_value)revenue
from order_items join payments 
on order_items.order_id = payments.order_id
group by order_items.seller_id)as a
"""

cur.execute(query)
data = cur.fetchall()

df = pd.DataFrame(data, columns =["seller_id", "revenue", "rank"])
df = df.head()
sns.barplot(x = "seller_id", y ="revenue", data = df)

plt.xticks(rotation = 90)
plt.show()


# # Calculate the moving average of order values for each customer over their order history.

# In[17]:


query = """
select customer_id, order_purchase_timestamp, payment,
avg(payment) over (partition by customer_id order by order_purchase_timestamp
rows between 2 preceding and current row) as mov_avg
from
(SELECT orders.customer_id, orders.order_purchase_timestamp,
payments.payment_value as payment
from payments join orders
on payments.order_id = orders.order_id)as a;

"""

cur.execute(query)
data = cur.fetchall()

df = pd.DataFrame(data)
df


# # Calculate the cumulative sales per month for each year.

# In[18]:


query = """
select years, months, payment, sum(payment)
over(order by years, months) cumulative_sales from
(SELECT 
    YEAR(orders.order_purchase_timestamp) AS years,
    MONTH(orders.order_purchase_timestamp) AS months,
    ROUND(SUM(payments.payment_value), 2) AS Payment
FROM orders
JOIN payments ON orders.order_id = payments.order_id
GROUP BY years, months
ORDER BY years, months) as a
"""

cur.execute(query)
data = cur.fetchall()

df = pd.DataFrame(data)
df


# # Calculate the year-over-year growth rate of total sales.

# In[25]:


query = """
with a as (select 
    YEAR(orders.order_purchase_timestamp) AS years,
    ROUND(SUM(payments.payment_value), 2) AS Payment
FROM orders
JOIN payments ON orders.order_id = payments.order_id

GROUP BY years ORDER BY years)

select years, (payment - lag(payment, 1) over ( order by years))/ 
lag(payment, 1) over ( order by years) *100 from a
"""

cur.execute(query)
data = cur.fetchall()

df = pd.DataFrame(data, columns =["Years", "Yoy % growth"])
df


# # Calculate the retention rate of customers, defined as the percentage of customers who make another purchase within 6 months of their first purchase.

# In[26]:


query = """
WITH a AS (
    SELECT 
        customers.customer_id,
        MIN(orders.order_purchase_timestamp) AS first_order
    FROM customers
    JOIN orders ON customers.customer_id = orders.customer_id
    GROUP BY customers.customer_id
),
b AS (
    SELECT 
        a.customer_id, 
        COUNT(DISTINCT orders.order_purchase_timestamp) AS order_count
    FROM a
    JOIN orders ON orders.customer_id = a.customer_id
        AND orders.order_purchase_timestamp > a.first_order
        AND orders.order_purchase_timestamp < DATE_ADD(a.first_order, INTERVAL 6
        MONTH)
    GROUP BY a.customer_id
)
SELECT 100 * (COUNT(DISTINCT a.customer_id) / COUNT(DISTINCT b.customer_id))
FROM a 
LEFT JOIN b ON a.customer_id = b.customer_id;
"""

cur.execute(query)
data = cur.fetchall()

data


# # Identify the top 3 customers who spent the most money in each year.
# 

# In[32]:


query = """
select years, customer_id, payment, d_rank
from
(select year(orders.order_purchase_timestamp)years,
orders.customer_id,
sum(payments.payment_value) payment,
dense_rank() over (partition by year(orders.order_purchase_timestamp)
order by sum(payments.payment_value)desc) d_rank
from orders join payments
on payments.order_id = orders.order_id
group by year(orders.order_purchase_timestamp),
orders.customer_id) as a
where d_rank <=3;
"""

cur.execute(query)
data = cur.fetchall()
df = pd.DataFrame(data, columns =["year", "id","payment", "rank"])
sns.barplot(x ="id", y ="payment", data = df, hue = "year")
plt.xticks(rotation = 90)
plt.show()

