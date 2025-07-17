# Exploring Frequent Itemsets: Closed vs Maximal in Supermarket Data

This notebook focuses on simulating transaction data for a supermarket scenario and applying frequent pattern mining using the Apriori algorithm.

---

## Objective

In this section, we simulate transaction data that will later be used to identify:

- **Frequent Itemsets**
- **Closed Frequent Itemsets**
- **Maximal Frequent Itemsets**

---

## Task 1: Simulate Supermarket Transaction Data

In this task, we simulate **3,000 supermarket transactions**.  
Each transaction contains between **2 and 7 items**, randomly selected from a pool of **30 unique items**.

The resulting dataset is saved to `supermarket_transactions.csv` for further analysis.

**Student Responsible**: Selmah Tzindori  
**Modified by**: Hana Gashaw (for reproducibility)

---

## Python Code

```python
import random
import pandas as pd

# -------------------------------
# Step 0: Set seed for reproducibility
# -------------------------------
# Ensures that random selections (bundles and extra items) are the same every time the script runs.
random.seed(42)

# -------------------------------
# Step 1: Define a pool of items
# -------------------------------
# A diverse list of 30+ common grocery items typically found in a supermarket.
item_pool = [
    'milk', 'bread', 'eggs', 'cheese', 'butter', 'juice', 'apples', 'bananas', 'oranges', 'grapes',
    'cereal', 'chocolate', 'yogurt', 'chicken', 'beef', 'pasta', 'rice', 'tomatoes', 'onions', 'potatoes',
    'carrots', 'lettuce', 'beans', 'soda', 'water', 'coffee', 'tea', 'cookies', 'ice cream', 'toilet paper'
]

# -------------------------------
# Step 2: Define common frequent bundles
# -------------------------------
# These are multi-item sets that will be intentionally injected into 50% of transactions
# to simulate real-world item associations (like milk+bread or apples+bananas+yogurt).
frequent_bundles = [
    ['milk', 'bread'],
    ['apples', 'bananas', 'yogurt'],
    ['chicken', 'rice', 'beans'],
    ['soda', 'chips', 'cookies'],  # Note: 'chips' will be added to item pool
    ['cheese', 'butter', 'eggs']
]

# Add any missing bundle items to item pool (e.g., 'chips' not in original pool)
item_pool = list(set(item_pool + ['chips']))

# -------------------------------
# Step 3: Generate synthetic transactions
# -------------------------------
# Loop generates 3,000 transactions. Each transaction:
# - Has a 50% chance of including one frequent bundle
# - Adds 0 to 4 extra random (non-duplicate) items
# - Randomizes item order to avoid fixed patterns
num_transactions = 3000
transactions = []

for _ in range(num_transactions):
    transaction = []

    # Inject a frequent bundle 50% of the time
    if random.random() < 0.5:
        bundle = random.choice(frequent_bundles)
        transaction.extend(bundle)

    # Add a few additional random items (avoid duplicates)
    num_extra_items = random.randint(0, 4)
    remaining_items = list(set(item_pool) - set(transaction))
    extras = random.sample(remaining_items, num_extra_items)
    transaction.extend(extras)

    # Shuffle items so the order is randomized
    random.shuffle(transaction)
    transactions.append(transaction)

# -------------------------------
# Step 4: Save transactions to CSV
# -------------------------------
# Each transaction is saved as a comma-separated string in one row.
# Useful for visual inspection or loading later.
transaction_strings = [', '.join(t) for t in transactions]
transactions_df = pd.DataFrame({'Transaction': transaction_strings})
transactions_df.to_csv('supermarket_transactions.csv', index=False)

# -------------------------------
# Step 5: Export CSV file
# -------------------------------
# This will create a CSV file named 'supermarket_transactions.csv' in the current directory. 
transactions_df.to_csv('supermarket_transactions.csv', index=False)

# -------------------------------
# Step 6: Preview the simulated data
# -------------------------------
# Display the first few rows to confirm structure and content.
print("Sample Transactions:")
transactions_df.head()
```

---

## Sample Output

```
Sample Transactions:
                                       Transaction
0                             cheese, butter, eggs
1                                           tea
2  chips, tomatoes, coffee, bread, ice cream, milk
3                    tea, eggs, soda, coffee
4                          lettuce, carrots
```

---

## Notes

- The dataset simulates **real-world supermarket transactions** with natural associations (frequent bundles) to help improve the performance and realism of pattern mining.
- The transactions are saved as strings for convenience and can be processed later into a basket format for mining algorithms.

---

# Task 2: Convert Transactions to One-Hot Encoded Format After Cleaning

In this task, we convert the simulated supermarket transaction data into a one-hot encoded format.

This transformation is necessary for applying the `apriori()` algorithm from the `mlxtend.frequent_patterns` module. One-hot encoding involves converting each transaction into a row, with items as columns and Boolean values (`True` or `False`) indicating whether an item is present in the transaction.

### Student Responsible: Levin Ekuam  
*Modified by: Hana Gashaw*

---

## 🔧 Step-by-Step Code with Explanation

```python
# Required libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd

# Step 1: Drop any rows where the transaction is missing (NaN or empty string)
transactions_df.dropna(subset=['Transaction'], inplace=True)
transactions_df = transactions_df[transactions_df['Transaction'].str.strip() != '']

# Step 2: Reset index after cleaning
transactions_df.reset_index(drop=True, inplace=True)

# Step 3: Convert the transaction strings to lists
transactions = transactions_df['Transaction'].apply(lambda x: x.strip().split(', '))
```

---

## Encoding Using TransactionEncoder

```python
# Step 4: Initialize the encoder
te = TransactionEncoder()

# Step 5: Fit and transform the transactions
te_ary = te.fit(transactions).transform(transactions)

# Step 6: Create the encoded DataFrame
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
```

---

## Output: One-Hot Encoded Transaction Data

```python
df_encoded.head()
```

**Sample Output:**

| apples | bananas | beans | beef | bread | butter | carrots | cereal | cheese | chicken | ... | oranges | pasta | potatoes | rice | soda | tea | toilet paper | tomatoes | water | yogurt |
|--------|---------|-------|------|--------|--------|----------|--------|--------|----------|-----|----------|--------|-----------|------|------|-----|----------------|-----------|--------|--------|
| False  | False   | False | False | False | False | False   | False | False | False   | ... | False   | False | False     | False | False | True | False         | False     | False | False  |
| False  | False   | False | False | True  | False | False   | False | False | False   | ... | False   | False | False     | False | False | False | False         | True      | False | False  |
| False  | False   | False | False | False | False | False   | False | False | False   | ... | False   | False | False     | False | True  | True | False         | False     | False | False  |
| False  | False   | False | False | False | False | True    | False | False | False   | ... | False   | False | False     | False | False | False | False         | False     | False | False  |
| False  | False   | False | False | False | False | False   | False | False | False   | ... | False   | False | False     | False | False | False | False         | True      | False | False  |

(*Note: Output is truncated for brevity; full dataset includes 30+ item columns and 3,000 rows.*)

---

**Conclusion:**  
We've successfully cleaned the transaction data and converted it into a one-hot encoded format. This DataFrame can now be used to apply frequent itemset mining algorithms such as Apriori to discover frequent, closed, and maximal itemsets.

# Task 3: Find Frequent Itemsets using the Apriori Algorithm

In this task, we apply the **Apriori algorithm** from the `mlxtend.frequent_patterns` module to identify **frequent itemsets**—combinations of items that appear together in transactions with a minimum support threshold of **5%** (0.05).

The algorithm returns all itemsets whose **support** (i.e., the proportion of transactions that contain the itemset) is at least 0.05. We then round the support values for readability and display the top 10 frequent itemsets based on this metric.

### Student Responsible: Ted Korir

---

## 🔧 Step-by-Step Code with Explanation

```python
from mlxtend.frequent_patterns import apriori

# ----------------------------------------
# Step 1: Generate frequent itemsets
# ----------------------------------------
# Using Apriori to extract frequent itemsets with minimum support of 0.05
# Setting use_colnames=True ensures the item names appear in the output
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
```

```python
# ----------------------------------------
# Step 2: Round support values
# ----------------------------------------
# The support column shows the proportion of transactions containing each itemset
# Rounding to 2 decimal places for clarity
frequent_itemsets['support'] = frequent_itemsets['support'].round(2)
```

```python
# ----------------------------------------
# Step 3: Display top 10 frequent itemsets
# ----------------------------------------
# Show the first 10 most frequent itemsets
print(frequent_itemsets.head(10))
```

```python
# Optional: Export top 10 itemsets to a CSV file for reporting
frequent_itemsets.head(10).to_csv('frequent_itemsets.csv', index=False)
```

---

## Output: Top 10 Frequent Itemsets

| support | itemsets     |
|---------|--------------|
| 0.17    | (apples)     |
| 0.17    | (bananas)    |
| 0.17    | (beans)      |
| 0.08    | (beef)       |
| 0.17    | (bread)      |
| 0.18    | (butter)     |
| 0.07    | (carrots)    |
| 0.08    | (cereal)     |
| 0.18    | (cheese)     |
| 0.17    | (chicken)    |

---

**Conclusion:**  
The Apriori algorithm successfully identified items (and item combinations) that occur frequently across transactions. These frequent itemsets can be used in subsequent steps such as generating association rules for recommendation systems or market basket analysis.

# Task: Find Closed Frequent Itemsets

In this task, we identify **Closed Frequent Itemsets** from the list of frequent itemsets generated earlier using the Apriori algorithm. A frequent itemset is considered **closed** if none of its **proper supersets** have the same support value. This helps in reducing redundancy while retaining all useful association information.

### Student Responsible: Angela Irungu

---

##  What is a Closed Frequent Itemset?

A **closed frequent itemset** is one where **no superset** of the itemset has **the same support**. Closed itemsets are useful because they compactly represent all frequent itemsets without losing any support information.

---

##  Step-by-Step Code with Explanation

```python
# NOTE: Assumes that 'frequent_itemsets' has already been generated 
# using the apriori() function

# Step 1: Initialize a list to hold closed itemsets
closed_itemsets = []

# Step 2: Loop through each frequent itemset and compare it with others
for i, row in frequent_itemsets.iterrows():
    current_itemset = row['itemsets']
    current_support = row['support']
    is_closed = True  # Start by assuming the itemset is closed

    # Step 3: Check for a proper superset with the same support
    for j, other_row in frequent_itemsets.iterrows():
        other_itemset = other_row['itemsets']
        other_support = other_row['support']

        # If another itemset is a proper superset and has equal support, current is not closed
        if current_itemset < other_itemset and current_support == other_support:
            is_closed = False
            break

    # Step 4: If no such superset is found, add to closed itemsets
    if is_closed:
        closed_itemsets.append((current_itemset, current_support))
```

```python
# Step 5: Convert closed itemsets to DataFrame for display and export
closed_df = pd.DataFrame(closed_itemsets, columns=["itemsets", "support"])

# Step 6: Export the closed itemsets to a CSV file (optional)
closed_df.to_csv("closed_itemsets.csv", index=False)

# Step 7: Display the result
print("Closed Frequent Itemsets:")
print(closed_df)
```

---

## Output: Closed Frequent Itemsets

Here are some of the resulting closed frequent itemsets:

| itemsets                 | support |
|--------------------------|---------|
| (apples)                 | 0.17    |
| (bananas)                | 0.17    |
| (beans)                  | 0.17    |
| (beef)                   | 0.08    |
| (bread)                  | 0.17    |
| (butter)                 | 0.18    |
| (carrots)                | 0.07    |
| (cereal)                 | 0.08    |
| (cheese)                 | 0.18    |
| (chicken)                | 0.17    |
| ...                      | ...     |
| (yogurt, apples)         | 0.11    |
| (milk, bread)            | 0.12    |
| (soda, cookies, chips)   | 0.13    |
| (eggs, cheese, butter)   | 0.11    |

>  **Note:** This table only shows a subset of the closed itemsets. The full list was exported to `closed_itemsets.csv`.

---

## Conclusion

Identifying **closed frequent itemsets** provides a more compact representation of frequent patterns in the data while preserving all essential support information. This step sets a strong foundation for rule generation and efficient pattern analysis in future steps.

# TASK 5: Maximal Frequent Itemsets

**Student Responsible: Trizah Nzioka**

---

## Overview

In this task, we identify **maximal frequent itemsets** from the itemsets already generated using the **Apriori algorithm**.

A **maximal frequent itemset** is one that is:

- Frequent (meets the minimum support threshold), **and**
- **Not** a subset of any other frequent itemset.

This means there’s no **larger itemset** (i.e., a superset) that is also frequent.

---

## How It Works

1. Loop through each itemset in the list of frequent itemsets.
2. For each one, check whether it is a subset of any **other** frequent itemset.
3. If such a **superset** exists, mark it as **not maximal**.
4. If no superset is found, it is **maximal** and gets added to the result list.
5. Save the final results to a CSV file (`maximal_itemsets.csv`).
6. Display the first few maximal frequent itemsets.

---

## Code

```python
# [Student: Trizah Nzioka] Find Maximal Frequent Itemsets

# Maximal frequent itemsets are those that:
# - Are frequent (appear in enough transactions, i.e., ≥ min_support)
# - Have no **frequent superset** (i.e., no larger itemset that is also frequent)

# Step 1: Create an empty list to hold all maximal itemsets
maximal_itemsets = []

# Step 2: Loop through each frequent itemset found using Apriori
for i, row in frequent_itemsets.iterrows():
    current_itemset = row['itemsets']  # The itemset under consideration
    is_maximal = True                  # Assume it's maximal unless proven otherwise

    # Step 3: Compare current_itemset with all other itemsets
    for j, other_row in frequent_itemsets.iterrows():
        other_itemset = other_row['itemsets']

        # Check if there's a **proper superset** of current_itemset
        # If yes, current_itemset is not maximal
        if current_itemset < other_itemset:
            is_maximal = False
            break   # No need to continue checking other itemsets

    # Step 4: If no frequent superset was found, add to maximal list
    if is_maximal:
        maximal_itemsets.append((current_itemset, row['support']))

# Step 5: Convert the list of maximal itemsets into a DataFrame
maximal_df = pd.DataFrame(maximal_itemsets, columns=["itemsets", "support"])

# Step 6: Save results to a CSV file (optional for reporting)
maximal_df.to_csv("maximal_itemsets.csv", index=False)

# Step 7: Print the first 5 maximal frequent itemsets for review
print("Maximal Frequent Itemsets (first 5):")
print(maximal_df)
```

---

## Output

```plaintext
Maximal Frequent Itemsets (first 5):
                     itemsets  support
0                      (beef)     0.08
1                   (carrots)     0.07
2                    (cereal)     0.08
3                 (chocolate)     0.07
4                    (coffee)     0.07
5                    (grapes)     0.08
6                 (ice cream)     0.07
7                     (juice)     0.07
8                   (lettuce)     0.08
9                    (onions)     0.07
10                  (oranges)     0.08
11                    (pasta)     0.07
12                 (potatoes)     0.08
13                      (tea)     0.07
14             (toilet paper)     0.07
15                 (tomatoes)     0.08
16                    (water)     0.07
17              (milk, bread)     0.12
18  (yogurt, apples, bananas)     0.10
19     (beans, chicken, rice)     0.10
20     (eggs, cheese, butter)     0.11
21     (soda, cookies, chips)     0.13
```

---


These are the **maximal frequent itemsets** because they are **not subsets** of any larger frequent itemset in the dataset.

# **Team Contributions Summary**

- **Selmah Tzindori** – *tzindori@gmail.com*  
  Responsible for **Task 1: Simulate Supermarket Transaction Data**

- **Hana Gashaw** – *21ibtj@gmail.com*  
  Contributed to **Task 1** by making the data generation reproducible, assisted in **Task 2** by cleaning missing values, and authored the **README file**. She also reviewed and ensured the consistency and accuracy of all sections.

- **Levin Ekuam** – *lekuam1@gmail.com*  
  Responsible for **Task 2: Convert Transactions to One-Hot Encoded Format**

- **Ted Korir** – *koechted18@gmail.com*  
  Responsible for **Task 3: Find Frequent Itemsets using the Apriori Algorithm**

- **Angela Irungu** – *Irunguangela05@gmail.com*  
  Responsible for **Task 4: Find Closed Frequent Itemsets**

- **Trizah Nzioka** – *tmnalma6@gmail.com*  
  Responsible for **Task 5: Maximal Frequent Itemsets**





