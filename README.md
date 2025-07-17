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
