import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    from transformers import pipeline
except ImportError:
    pipeline = None
class DataAnalysis:
    def __init__(self, data):
        self.df = data

    def list_suitable_variables(self, test_type):
        print(f"Available variables for {test_type}:")
        if test_type == 'ANOVA':
            print("Continuous (interval/ratio) variables:")
            for var in self.df.select_dtypes(include=['int64', 'float64']).columns:
                print(var)
            print("Categorical (ordinal/nominal) variables:")
            for var in self.df.select_dtypes(include=['object']).columns:
                if self.df[var].nunique() > 1 and self.df[var].nunique() < 10:  
                    print(var)
        elif test_type == 'Regression':
            print("Continuous (interval/ratio) variables:")
            for var in self.df.select_dtypes(include=['int64', 'float64']).columns:
                print(var)
        elif test_type == 'Chi-Square':
            print("Categorical (nominal/ordinal) variables:")
            for var in self.df.select_dtypes(include=['object']).columns:
                if self.df[var].nunique() > 1 and self.df[var].nunique() < 10:
                    print(var)
        elif test_type == 't-Test':
            print("Continuous (interval/ratio) variables:")
            for var in self.df.select_dtypes(include=['int64', 'float64']).columns:
                print(var)

    def check_normality(self, var):
        try:
            stat, p_value = ss.shapiro(self.df[var].dropna())
            sns.histplot(self.df[var].dropna(), kde=True)
            plt.title(f"Normality Check for {var} (p-value={p_value:.4f})")
            plt.show()
            return p_value > 0.05
        except Exception as e:
            print("Error checking normality:", e)
            return False

    def anova_test(self, continuous_var, categorical_var):
        print(f"Performing ANOVA for '{continuous_var}' and '{categorical_var}'...")
        try:
            normal_dist = self.check_normality(continuous_var)
            self.plot_qq_histogram(continuous_var)
            if not normal_dist:
                print(f"'{continuous_var}' is not normally distributed, performing Kruskal-Wallis Test...")
                self.kruskal_wallis_test(continuous_var, categorical_var)
            else:
                f_value, p_value = ss.f_oneway(*(self.df[self.df[categorical_var] == level][continuous_var]
                                                 for level in self.df[categorical_var].unique()))
                print(f"ANOVA Result:\nF-value: {f_value}\np-value: {p_value}")
                if p_value < 0.05:
                    print("Null Hypothesis Rejected: There is a significant difference.")
                else:
                    print("Failed to Reject Null Hypothesis: No significant difference.")
        except Exception as e:
            print("Error performing ANOVA test:", e)

    def kruskal_wallis_test(self, continuous_var, categorical_var):
        try:
            stat, p_value = ss.kruskal(*(self.df[self.df[categorical_var] == level][continuous_var]
                                         for level in self.df[categorical_var].unique()))
            print(f"Kruskal-Wallis Result:\nStatistic: {stat}\np-value: {p_value}")
            if p_value < 0.05:
                print("Null Hypothesis Rejected: Statistically significant difference.")
            else:
                print("Failed to Reject Null Hypothesis.")
        except Exception as e:
            print("Error performing Kruskal-Wallis test:", e)

    def t_test(self, group1, group2):
        try:
            normal_dist1 = self.check_normality(group1)
            normal_dist2 = self.check_normality(group2)
            self.plot_qq_histogram(group1)
            self.plot_qq_histogram(group2)
            if normal_dist1 and normal_dist2:
                t_stat, p_value = ss.ttest_ind(self.df[group1].dropna(), self.df[group2].dropna())
                print(f"t-Test Result:\nt-statistic: {t_stat}\np-value: {p_value}")
                if p_value < 0.05:
                    print("Null Hypothesis Rejected: Statistically significant difference.")
                else:
                    print("Failed to Reject Null Hypothesis.")
            else:
                print("Performing Mann-Whitney U Test due to non-normal distribution...")
                u_stat, p_value = ss.mannwhitneyu(self.df[group1].dropna(), self.df[group2].dropna())
                print(f"Mann-Whitney U Result:\nU-statistic: {u_stat}\np-value: {p_value}")
        except Exception as e:
            print("Error performing t-Test:", e)

    def chi_square_test(self, categorical_var1, categorical_var2):
        try:
            contingency_table = pd.crosstab(self.df[categorical_var1], self.df[categorical_var2])
            chi2, p_value, _, _ = ss.chi2_contingency(contingency_table)
            print(f"Chi-Square Test Result:\nChi2 Statistic: {chi2}\np-value: {p_value}")
            if p_value < 0.05:
                print("Null Hypothesis Rejected: There is a statistically significant relationship.")
            else:
                print("Failed to Reject Null Hypothesis.")
        except Exception as e:
            print("Error performing Chi-Square test:", e)

    def regression_analysis(self, dependent_var, independent_vars):
        try:
            X = self.df[independent_vars]
            Y = self.df[dependent_var]
            self.check_normality(dependent_var)
            self.check_normality(independent_vars)
            slope, intercept, r_value, p_value, std_err = ss.linregress(X, Y)
            plt.scatter(X, Y, label='Data points')
            plt.plot(X, intercept + slope*X, 'r', label='Fitted line')
            print(f"""Slope is : {slope}
Intercept is : {intercept}
R_value is : {r_value}
Std_err is : {std_err}
P_value is : {p_value}
""")
            if p_value < 0.05:
                print("Null Hypothesis Rejected: There is a significant difference.")
            else:
                print("Failed to Reject Null Hypothesis: No significant difference.")
        except Exception as e:
            print("Error performing regression analysis:", e)

    def get_variable_types(self):
        variable_types = {}
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() > 1 and self.df[col].nunique() < 10:
                    variable_types[col] = 'Categorical'
                else:
                    variable_types[col] = 'Text'
            elif self.df[col].dtype in ['int64', 'float64']:
                variable_types[col] = 'Continuous'
            else:
                variable_types[col] = 'Other'
        return variable_types
    def plot_qq_histogram(self, title):
        """Plot Q-Q plot and histogram for the chosen variable."""
        data = self.df[title]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        ss.probplot(data, dist="norm", plot=plt)
        plt.title(f"Q-Q plot of {title}")
        plt.show()  
    def get_text_columns(self):
        
        text_columns = [col for col in self.df.columns if 'remark' in col]
        column_data = []
        
        for col in text_columns:
            avg_length = self.df[col].apply(len).mean()
            unique_entries = len(self.df[col].unique())
            column_data.append({
                'Column Name': col,
                'Average Entry Length': avg_length,
                'Unique Entries': unique_entries
            })
        
        return pd.DataFrame(column_data)

    def vader_sentiment_analysis(self, data):
        analyzer = SentimentIntensityAnalyzer()
        scores = data.apply(lambda remark: analyzer.polarity_scores(remark)['compound'])
        sentiments = data.apply(lambda remark: 'Positive' if analyzer.polarity_scores(remark)['compound'] >= 0 else 'Negative')
        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        scores = data.apply(lambda remark: TextBlob(remark).sentiment.polarity)
        sentiments = data.apply(lambda remark: 'Positive' if TextBlob(remark).sentiment.polarity >= 0 else 'Negative')
        subjectivity = data.apply(lambda remark: TextBlob(remark).sentiment.subjectivity)
        return scores, sentiments, subjectivity

    def distilbert_sentiment_analysis(self, data):
        if pipeline is not None:
            nlp = pipeline('sentiment-analysis')
            scores = data.apply(lambda remark: nlp(remark)[0]['score'])
            sentiments = data.apply(lambda remark: nlp(remark)[0]['label'])
            return scores, sentiments
        else:
            print("DistilBERT is not available.")
            return None, None