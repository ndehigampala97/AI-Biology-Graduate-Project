import json
import numpy as np
import pandas as pd

# =========================
# User settings
# =========================
INPUT_CSV = 'g2f_2023_phenotypic_clean_data.csv'
OUTPUT_CSV = 'g2f_2023_phenotypic_model_ready.csv'
TARGET = 'Grain Yield (bu/A)'

DROP_COLS = [
    'Filler',
    'Comments',
    "Plot Discarded [enter 'yes' or blank]",
    'Plot_ID',
    'Plot',
    'Range',
    'Pass',
    'Tester',
    'Snap [# of plants]',
    'Silking [MM/DD/YY]',
    'Anthesis [MM/DD/YY]',
    'Date Plot Harvested [MM/DD/YY]',
    'Date Plot Planted [MM/DD/YY]'
]

# =========================
# Load raw data
# =========================
df = pd.read_csv(INPUT_CSV, encoding='latin1', low_memory=False)
print('Raw shape:', df.shape)

# Drop rows where target is missing
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in the CSV.")

df = df.dropna(subset=[TARGET]).copy()
print('Shape after dropping rows with missing target:', df.shape)

# Drop selected columns if they exist
drop_cols = [c for c in DROP_COLS if c in df.columns]
df = df.drop(columns=drop_cols)
print('Dropped columns:', drop_cols)
print('Shape after dropping selected columns:', df.shape)

# Separate feature columns for filling missing values
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Missing summary before filling
missing_before = X.isna().sum().sort_values(ascending=False)
summary = pd.DataFrame({
    'column': missing_before.index,
    'missing_before_fill': missing_before.values
})

# Fill numeric columns with median of full dataset
num_fill_values = X[numeric_cols].median()
X[numeric_cols] = X[numeric_cols].fillna(num_fill_values)

# Fill categorical columns with mode of full dataset
cat_fill_values = {}
for col in categorical_cols:
    mode_vals = X[col].mode(dropna=True)
    fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 'Missing'
    cat_fill_values[col] = fill_val
    X[col] = X[col].fillna(fill_val)

# Combine back target
clean_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

# Missing summary after filling
summary['missing_after_fill'] = clean_df.drop(columns=[TARGET]).isna().sum().reindex(summary['column']).values
summary.to_csv('missing_value_summary_after_cleaning.csv', index=False)

# Save cleaned CSV
clean_df.to_csv(OUTPUT_CSV, index=False)

# Save fill values for record
fill_values_json = {
    'input_csv': INPUT_CSV,
    'output_csv': OUTPUT_CSV,
    'target': TARGET,
    'dropped_columns': drop_cols,
    'numeric_fill_values': {k: (None if pd.isna(v) else float(v)) for k, v in num_fill_values.items()},
    'categorical_fill_values': cat_fill_values,
}
with open('fill_values_used_cleaning_only.json', 'w') as f:
    json.dump(fill_values_json, f, indent=2)

print('\nSaved files:')
print(f'- {OUTPUT_CSV}')
print('- missing_value_summary_after_cleaning.csv')
print('- fill_values_used_cleaning_only.json')
print('\nFinal shape:', clean_df.shape)
print('Total missing values left:', clean_df.isna().sum().sum())
