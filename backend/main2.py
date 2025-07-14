import pandas as pd
import os

os.makedirs("test_data", exist_ok=True)

def generate_loan_files():
    loans = pd.DataFrame({
        "LoanID": ["L001", "L002", "L003", "L004", "L005"],
        "CustomerName": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "LoanAmount": [5000, 10000, 7500, 6000, 9000],
    })

    payments = pd.DataFrame({
        "ID": ["L001", "L002", "L004", "L006"],
        "AmountPaid": [5000, 7500, 6000, 1000],
    })

    loans.to_csv("test_data/loan_master.csv", index=False)
    payments.to_csv("test_data/loan_payments.csv", index=False)

def generate_bank_recon_files():
    internal = pd.DataFrame({
        "TransactionID": ["T001", "T002", "T003", "T004"],
        "Date": ["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04"],
        "Amount": [1500, -200, -50, 3000],
    })

    bank = pd.DataFrame({
        "ReferenceID": ["T001", "T003", "T005"],
        "Date": ["2025-07-01", "2025-07-03", "2025-07-05"],
        "Amount": [1500, -50, -100],
    })

    internal.to_csv("test_data/internal_ledger.csv", index=False)
    bank.to_csv("test_data/bank_statement.csv", index=False)

def generate_sales_inventory_files():
    sales = pd.DataFrame({
        "ProductID": ["P001", "P002", "P003", "P004"],
        "UnitsSold": [10, 20, 5, 12],
    })

    inventory = pd.DataFrame({
        "SKU": ["P001", "P003", "P004", "P005"],
        "StockRemaining": [90, 45, 38, 100],
    })

    sales.to_csv("test_data/sales.csv", index=False)
    inventory.to_csv("test_data/inventory.csv", index=False)

def generate_unrelated_dummy_files():
    df1 = pd.DataFrame({
        "Age": [25, 32, 41, 28],
        "Score": [88.5, 92.0, 76.5, 85.0],
        "City": ["Delhi", "Mumbai", "Chennai", "Pune"]
    })

    df2 = pd.DataFrame({
        "UserID": [1001, 1002, 1003],
        "Rating": [4.2, 3.8, 4.9],
        "Feedback": ["Good", "Average", "Excellent"]
    })

    df1.to_csv("test_data/unrelated1.csv", index=False)
    df2.to_csv("test_data/unrelated2.csv", index=False)


def generate_employee_salary_files():
    hr = pd.DataFrame({
        "EmpID": [101, 102, 103, 104],
        "Name": ["Asha", "Rahul", "Meera", "Zubin"]
    })

    payroll = pd.DataFrame({
        "EmployeeID": [101, 102, 105],
        "SalaryPaid": [50000, 52000, 48000]
    })



    hr.to_csv("test_data/hr_data.csv", index=False)
    payroll.to_csv("test_data/payroll.csv", index=False)

generate_loan_files()
generate_bank_recon_files()
generate_sales_inventory_files()
generate_employee_salary_files()
generate_unrelated_dummy_files()

print("✅ Dummy test file pairs generated in /test_data")
