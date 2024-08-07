import os
import pandas as pd

# Directory containing the CSV files
directory = 'combined/combined'

# Load baseline RMSE
baseline_df = pd.read_csv(os.path.join(directory, 'none_epoch_metrics.csv'))
baseline_rmse = baseline_df['rmse'].mean()

# Initialize a dictionary to hold the data
data = {'Feature': [], 'RMSE': [], 'Indicator': []}

# Process each file in the directory
for filename in os.listdir(directory):
    if "_epoch_metrics.csv" in filename:
        df = pd.read_csv(os.path.join(directory, filename))
        avg_rmse = df['rmse'].mean().round(1)
        improvement = (avg_rmse - baseline_rmse) / baseline_rmse * 100
        feature_name = filename.split('_epoch_metrics.csv')[0]
        data['Feature'].append(feature_name)
        data['RMSE'].append(avg_rmse)
        if avg_rmse > baseline_rmse:
            indicator = r"\textcolor{red}{$\uparrow$} " + f"{abs(improvement):.1f}\%"
        elif avg_rmse < baseline_rmse:
            indicator = r"\textcolor{green}{$\downarrow$} " + f"{abs(improvement):.1f}\%"
        else:
            indicator = r"\textcolor{black}{=}"
        data['Indicator'].append(indicator)

# Create a DataFrame from the dictionary
results_df = pd.DataFrame(data)

# Sort the dataframe to have baseline at the top
results_df = results_df.sort_values(by='Feature', key=lambda x: x != 'none').reset_index(drop=True)
print(results_df)

# Generate LaTeX table code
latex_code = r"\begin{tabular}{ll}"
latex_code += r"\hline"
latex_code += r"Feature & RMSE & Indicator \\"
latex_code += r"\hline"
for _, row in results_df.iterrows():
    latex_code += f"{row['Feature']} & {row['RMSE']:.1f} {row['Indicator']} \\"
latex_code += r"\hline"
latex_code += r"\end{tabular}"

print(latex_code)

