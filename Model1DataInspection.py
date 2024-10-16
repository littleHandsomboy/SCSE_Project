import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
class DataInspection:
    def __init__(self):
        self.df = None 

    def load_csv(self, file_path):
        self.df = pd.read_csv(file_path)

    def drop_null_rows(self):
        self.df.dropna(inplace=True)
        print("Rows with null values have been dropped.")

    def plot_histogram(self, col):
        self.df[col].hist()
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    def plot_boxplot(self, col):
        self.df.boxplot(column=col)
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.ylabel('Values')
        plt.show()

    def plot_bar_chart(self, col):
        self.df[col].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

    def plot_scatter(self, x_col, y_col):
        self.df.plot.scatter(x=x_col, y=y_col)
        plt.title(f'Scatter Plot of {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def handle_missing_values(self, col):
        missing_percent = self.df[col].isna().mean() * 100
        if missing_percent > 50:
            self.df.drop(columns=[col], inplace=True)
            return False
        else:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            return True

    def check_data_types(self, col):
        if pd.api.types.is_string_dtype(self.df[col]):
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except ValueError:
                pass

    def classify_and_calculate(self, col):
        if not self.handle_missing_values(col):
            return None
        self.check_data_types(col)
        unique_count = self.df[col].nunique()
        if pd.api.types.is_numeric_dtype(self.df[col]):
            if unique_count < 10:
                result = self.df[col].median()
                self.plot_boxplot(col)
            else:
                result = self.df[col].mean()
                self.plot_histogram(col)
        else:
            result = self.df[col].mode()[0]
            self.plot_bar_chart(col)
        return result

    def classify_columns(self):
        for col in self.df.columns:
            self.classify_and_calculate(col)

    def ask_for_scatterplot(self):
        numeric_cols = self.numeric_columns()
        if len(numeric_cols) >= 2:
            print(f"Available numeric columns: {numeric_cols}")
            col1 = input("Choose first column: ")
            col2 = input("Choose second column: ")
            self.plot_scatter(numeric_cols[int(col1)-1], numeric_cols[int(col2)-1])

    def ask_for_boxplot(self):
        numeric_cols = self.numeric_columns()
        print(f"Available numeric columns: {numeric_cols}")
        col = input("Choose column for box plot: ")
        self.plot_boxplot(numeric_cols[int(col)-1])

    def ask_for_bar_chart(self):
        cols = self.df.columns
        print(f"Available columns: {cols}")
        col = input("Choose column for bar chart: ")
        self.plot_bar_chart(cols[int(col)-1])

    def ask_for_histogram(self):
        numeric_cols = self.numeric_columns()
        print(f"Available numeric columns: {numeric_cols}")
        col = input("Choose column for histogram: ")
        self.plot_histogram(numeric_cols[int(col)-1])

    def numeric_columns(self):
        return [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
    def summarize_columns(self):
        print(f"{'Variable':<15}{'Type':<10}{'Mean / Median / Mode':<20}{'Kurtosis':<10}{'Skewness':<10}")
        for col in self.df.columns:
            var_type = self.classify_variable_type(col)
            measure = self.calculate_measure(col, var_type)
            kurt = self.calculate_kurtosis(col) if var_type != 'Nominal' else 'NA'
            skewness = self.calculate_skewness(col) if var_type != 'Nominal' else 'NA'
            print(f"{col:<15}{var_type:<10}{measure:<20}{kurt:<10}{skewness:<10}")

    def classify_variable_type(self, col):
        if pd.api.types.is_numeric_dtype(self.df[col]):
            unique_count = self.df[col].nunique()
            if unique_count < 10:
                return 'Ordinal'
            else:
                return 'Ratio'
        else:
            return 'Nominal'

    def calculate_measure(self, col, var_type):
        if var_type == 'Nominal':
            mode_value = self.df[col].mode()[0]
            return str(mode_value)
        elif var_type == 'Ordinal':
            median_value = self.df[col].median()
            return f"{median_value:.2f}"
        else:  # Ratio
            mean_value = self.df[col].mean()
            return f"{mean_value:.2f}"

    def calculate_kurtosis(self, col):
        kurt = kurtosis(self.df[col].dropna())
        return f"{kurt:.2f}"

    def calculate_skewness(self, col):
        skewness = skew(self.df[col].dropna())
        return f"{skewness:.2f}"