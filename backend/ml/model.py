#Reconclation logic

# import pandas as pd
# loans = pd.read_csv("C:/Users/shree/Desktop/loan_master_full.csv")
# payments = pd.read_csv("C:/Users/shree/Desktop/loan_payment_full.csv")

# paid_summary = payments.groupby('LoanID')['AmountPaid'].sum().reset_index()

# # Merge loan info with payments
# merged = pd.merge(loans, paid_summary, on="LoanID", how="outer")

# # Fill NaN where payments are missing
# merged['AmountPaid'].fillna(0, inplace=True)

# # Add result column
# merged['Status'] = merged.apply(
#     lambda row: 'Fully Paid' if row['AmountPaid'] == row['LoanAmount']
#     else ('Partially Paid' if row['AmountPaid'] > 0 else 'No Payment'),
#     axis=1
# )

# # Save or return results
# data=(merged[['LoanID', 'CustomerName', 'LoanAmount', 'AmountPaid', 'Status']])
# print(data)


import pandas as pd
from fastapi import UploadFile

async def reconcile_files(file1: UploadFile, file2: UploadFile):
    loans = pd.read_csv(file1.file)
    payments = pd.read_csv(file2.file)

    paid_summary = payments.groupby('LoanID')['AmountPaid'].sum().reset_index()
    merged = pd.merge(loans, paid_summary, on="LoanID", how="outer")
    merged['AmountPaid'].fillna(0, inplace=True)

    merged['Status'] = merged.apply(
        lambda row: 'Fully Paid' if row['AmountPaid'] == row['LoanAmount']
        else ('Partially Paid' if row['AmountPaid'] > 0 else 'No Payment'),
        axis=1
    )

    result = merged[['LoanID', 'CustomerName', 'LoanAmount', 'AmountPaid', 'Status']]
    return result.to_dict(orient='records')


#........................................
#BAR

''' import matplotlib.pyplot as plt

# Clean NaN rows
clean_data = merged.dropna(subset=['LoanAmount', 'AmountPaid'])

# Bar chart
plt.figure(figsize=(8, 5))
plt.bar(clean_data['LoanID'], clean_data['LoanAmount'], label='Loan Amount', alpha=0.6)
plt.bar(clean_data['LoanID'], clean_data['AmountPaid'], label='Amount Paid', alpha=0.6)
plt.xlabel("Loan ID")
plt.ylabel("Amount (INR)")
plt.title("Loan Amount vs Amount Paid")
plt.legend()
plt.tight_layout()
plt.show()

#............................................
#pie chart
# Count status values
status_counts = merged['Status'].value_counts()

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Loan Payment Status Distribution")
plt.axis('equal')
pichart= plt.show() 
'''



