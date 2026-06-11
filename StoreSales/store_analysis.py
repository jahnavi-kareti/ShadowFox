import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# STEP 1: Load Dataset
df = pd.read_csv('StoreSales/Sample - Superstore.csv', encoding='latin1')
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
# STEP 2: Data Preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date']  = pd.to_datetime(df['Ship Date'])
df['Year']       = df['Order Date'].dt.year
df['Month']      = df['Order Date'].dt.month
df['Month Name'] = df['Order Date'].dt.strftime('%b')

print("\nMissing values:", df.isnull().sum().sum())
print("Date range:", df['Order Date'].min(), "to", df['Order Date'].max())
# STEP 3: Overall Summary
print("\n📊 Overall Business Summary:")
print(f"  Total Sales  : ${df['Sales'].sum():,.2f}")
print(f"  Total Profit : ${df['Profit'].sum():,.2f}")
print(f"  Profit Margin: {(df['Profit'].sum()/df['Sales'].sum()*100):.2f}%")
print(f"  Total Orders : {df['Order ID'].nunique()}")
print(f"  Total Customers: {df['Customer ID'].nunique()}")
# STEP 4: Sales & Profit by Category
category = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(category['Category'], category['Sales'],
            color=['#2196F3', '#4CAF50', '#FF9800'])
axes[0].set_title('Total Sales by Category', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Sales ($)')
for i, v in enumerate(category['Sales']):
    axes[0].text(i, v + 1000, f'${v:,.0f}', ha='center', fontsize=9)

axes[1].bar(category['Category'], category['Profit'],
            color=['#2196F3', '#4CAF50', '#FF9800'])
axes[1].set_title('Total Profit by Category', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Profit ($)')
for i, v in enumerate(category['Profit']):
    axes[1].text(i, v + 500, f'${v:,.0f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('StoreSales/sales_profit_by_category.png')
plt.show()
print("Saved: sales_profit_by_category.png")
# STEP 5: Sales & Profit by Region
region = df.groupby('Region')[['Sales', 'Profit']].sum().reset_index()
region['Profit Margin %'] = (region['Profit'] / region['Sales'] * 100).round(2)
print("\n📍 Sales & Profit by Region:")
print(region.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(region))
width = 0.35
bars1 = ax.bar(x - width/2, region['Sales'],  width, label='Sales',  color='#2196F3')
bars2 = ax.bar(x + width/2, region['Profit'], width, label='Profit', color='#4CAF50')
ax.set_title('Sales & Profit by Region', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(region['Region'])
ax.set_ylabel('Amount ($)')
ax.legend()
plt.tight_layout()
plt.savefig('StoreSales/sales_profit_by_region.png')
plt.show()
print("Saved: sales_profit_by_region.png")
# STEP 6: Monthly Sales Trend
monthly = df.groupby(['Year', 'Month'])[['Sales', 'Profit']].sum().reset_index()
monthly['Date'] = pd.to_datetime(monthly[['Year', 'Month']].assign(day=1))
monthly = monthly.sort_values('Date')

plt.figure(figsize=(14, 5))
plt.plot(monthly['Date'], monthly['Sales'],  marker='o', label='Sales',
         color='#2196F3', linewidth=2)
plt.plot(monthly['Date'], monthly['Profit'], marker='o', label='Profit',
         color='#4CAF50', linewidth=2)
plt.title('Monthly Sales & Profit Trend', fontsize=13, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('StoreSales/monthly_trend.png')
plt.show()
print("Saved: monthly_trend.png")
# STEP 7: Top 10 Products by Profit
top_products = df.groupby('Product Name')['Profit'].sum().nlargest(10).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='Profit', y='Product Name', data=top_products, palette='Greens_r')
plt.title('Top 10 Products by Profit', fontsize=13, fontweight='bold')
plt.xlabel('Total Profit ($)')
plt.tight_layout()
plt.savefig('StoreSales/top10_products.png')
plt.show()
print("Saved: top10_products.png")
# STEP 8: Discount vs Profit Analysis
plt.figure(figsize=(8, 5))
plt.scatter(df['Discount'], df['Profit'], alpha=0.3, color='#FF5722')
plt.title('Discount vs Profit', fontsize=13, fontweight='bold')
plt.xlabel('Discount')
plt.ylabel('Profit ($)')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('StoreSales/discount_vs_profit.png')
plt.show()
print("Saved: discount_vs_profit.png")
# STEP 9: Sub-Category Profit Analysis
subcat = df.groupby('Sub-Category')['Profit'].sum().sort_values()

colors = ['#F44336' if x < 0 else '#4CAF50' for x in subcat.values]
plt.figure(figsize=(12, 7))
subcat.plot(kind='barh', color=colors)
plt.title('Profit by Sub-Category (Red = Loss)', fontsize=13, fontweight='bold')
plt.xlabel('Total Profit ($)')
plt.axvline(x=0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('StoreSales/subcat_profit.png')
plt.show()
print("Saved: subcat_profit.png")
# STEP 10: Segment Analysis
segment = df.groupby('Segment')[['Sales', 'Profit']].sum().reset_index()
print("\n👥 Sales & Profit by Segment:")
print(segment.to_string(index=False))

plt.figure(figsize=(8, 5))
x = np.arange(len(segment))
width = 0.35
plt.bar(x - width/2, segment['Sales'],  width, label='Sales',  color='#9C27B0')
plt.bar(x + width/2, segment['Profit'], width, label='Profit', color='#E91E63')
plt.xticks(x, segment['Segment'])
plt.title('Sales & Profit by Customer Segment', fontsize=13, fontweight='bold')
plt.ylabel('Amount ($)')
plt.legend()
plt.tight_layout()
plt.savefig('StoreSales/segment_analysis.png')
plt.show()
print("Saved: segment_analysis.png")

print("\n🎉 Store Sales & Profit Analysis Complete!")
print(f"Total of 7 charts saved in StoreSales/ folder")