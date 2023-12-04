#12151389 Park Byung Moon 

import pandas as pd

data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

for year in range(2015, 2019):
	year_data = data_df[data_df['year'] == year]
    
	print(f"\nTop 10 Players in {year}:")
    	hits_top10 = year_data.nlargest(10, 'H')
    	print("Hits:\n", hits_top10[['batter_name', 'H']])
    
    	avg_top10 = year_data.nlargest(10, 'avg')
    	print("\nBatting Average:\n", avg_top10[['batter_name', 'avg']])
    
    	hr_top10 = year_data.nlargest(10, 'HR')
    	print("\nHome Runs:\n", hr_top10[['batter_name', 'HR']])
    
    	obp_top10 = year_data.nlargest(10, 'OBP')
    	print("\nOn-Base Percentage:\n", obp_top10[['batter_name', 'OBP']])

year_2018_data = data_df[data_df['year'] == 2018]
highest_war_by_position = year_2018_data.loc[year_2018_data.groupby('cp')['war'].idxmax()]
print("\nPlayer with the Highest WAR by Position in 2018:\n", highest_war_by_position[['batter_name', 'cp', 'war']])

selected_columns = ["R", "H", "HR", "RBI", "SB", "war", "avg", "OBP", "SLG", "salary"]
correlations = data_df[selected_columns].corr()

highest_correlation_feature = correlations['salary'].idxmax()
highest_correlation_value = correlations.loc[highest_correlation_feature, 'salary']

print(f"\nFeature with the Highest Correlation with Salary: {highest_correlation_feature} (Correlation: {highest_correlation_value})")