import Model1DataInspection as M1
import  Model2DataAnalysis as M2
try:
    from transformers import pipeline
except ImportError:
    pipeline = None
"Load data"

print("Welocme to Our Data Analysis Project!")
path = input("Give your file path here: ")
# path = "Datasets\CleanedData.csv"
# path="Datasets\TestSentimentAnalysisSet.csv"
# path = "Datasets\TestTsets.csv"
analysis = M1.DataInspection()
analysis.load_csv(path)
data = analysis.df
data_analysis = M2.DataAnalysis(analysis.df)
while(True):
    menu_num = input('''Welocme To Main Menu Choose Your Option Here:
    1.Clean Data
    2.Show Data Basic Inforamtion (Variable statistics is here)
    3.Show Data Inspection (Variable distribution is here)
    4.Data Analysis
    5.Exit 
    Give your choise:''')
    
    if(menu_num=='1'):
        analysis.drop_null_rows()
        print("Now Every Rows Where Hava NULL Values Has Been Deleted!")
        continue
    elif(menu_num=='2'):
        while(True):
            info_num = input('''Welcome to Data Basic Inforamtion Part:
    1.*Summarize columns* Variable statistics is here
    2.Show Data Information 
    3.Other Infomations
    4.Show Data Description
    5.Back Main Menu
    Give your choise:''')
            if(info_num=="1"):
                analysis.summarize_columns()
                
            elif(info_num=="2"):
                print(data.info())
            elif(info_num=="3"):
                print("Data Head 10 valuse:")
                print(data.head(10))

                print("Data Shep:")
                print(data.shape)

                print("Null valuse of data:")
                print(data.isna().sum())
            elif(info_num=="4"):
                print(data.describe().to_string)
            elif(info_num=="5"):
                break
            else:
                print("******Error Input******")
                print("Please give the right number, Try angin!")
                continue
    elif(menu_num=='3'):
        while(True):
            inspection_num = input("""Welcome to Data Inspection Part:
    1. Plot Histogram
    2. Plot Boxplot
    3. Plot Bar Chart
    4. Plot Scatter Plot
    5. Plot dominant                        
    6. Back to Main Menu
    Choose The Plot You Want to Polt: """)
            if(inspection_num == '1'):
                analysis.ask_for_histogram()
            elif(inspection_num == '2'):
                analysis.ask_for_boxplot()
            elif(inspection_num == '3'):
                analysis.ask_for_bar_chart()
            elif(inspection_num == '4'):
                analysis.ask_for_scatterplot()
            elif(inspection_num == '5'):
                analysis.plot_dominant()
            elif(inspection_num =='6'):
                break
            
            else:
                print("Invalid input. Please choose a valid option.")
    elif(menu_num=='4'):
        while(True):
            analysis_num = input("""
Welcome to Data Analysis Part: 
    1. Conduct ANOVA/Kruskal-Wallis
    2. Conduct t-Test/Mann-Whitney U
    3. Conduct Chi-Square
    4. Conduct Sentiment Analysis
    5. Conduct Regression
    6. Back to Main Menu
    Choose The Analysis Test You Want to Perform: """)

            if analysis_num == '1':
                data_analysis.list_suitable_variables('ANOVA')
                cont_var = input("Enter a continuous (interval/ratio) variable: ")
                cat_var = input("Enter a categorical (ordinal/nominal) variable: ")
                if cont_var in data_analysis.df.columns and cat_var in data_analysis.df.columns:
                    data_analysis.anova_test(cont_var, cat_var)
                else:
                    print("One or both variables are not suitable for the test.")
                analysis.plot_anova_boxplot(data,cont_var,cat_var,"ANOVA/KW")
            elif analysis_num == '2':
                data_analysis.list_suitable_variables('t-Test')
                group1 = input("Enter the first group variable: ")
                group2 = input("Enter the second group variable: ")
                if group1 in data_analysis.df.columns and group2 in data_analysis.df.columns:
                    data_analysis.t_test(group1, group2)
                else:
                    print("One or both variables are not suitable for the test.")

            elif analysis_num == '3':
                data_analysis.list_suitable_variables('Chi-Square')
                cat_var1 = input("Enter the first categorical variable: ")
                cat_var2 = input("Enter the second categorical variable: ")
                if cat_var1 in data_analysis.df.columns and cat_var2 in data_analysis.df.columns:
                    data_analysis.chi_square_test(cat_var1, cat_var2)
                else:
                    print("One or both variables are not suitable for the test.")
                analysis.plot_categorical_variables(cat_var1,cat_var2,"Chi-Square")
            elif analysis_num == '4':
                text_columns = data_analysis.get_text_columns()
                print(text_columns)
                column_choice = input("Enter the column to analyze: ")
                analysis_choice = int(input("Choose analysis method (1: Vader, 2: TextBlob, 3: DistilBERT): "))
                if analysis_choice == 1:
                    scores, sentiments = data_analysis.vader_sentiment_analysis(data_analysis.df[column_choice])
                    print(scores, sentiments)
                elif analysis_choice == 2:
                    scores, sentiments, subjectivity = data_analysis.textblob_sentiment_analysis(data_analysis.df[column_choice])
                    print(scores, sentiments, subjectivity)
                elif analysis_choice == 3 and pipeline is not None:
                    scores, sentiments = data_analysis.distilbert_sentiment_analysis(data_analysis.df[column_choice])
                    print(scores, sentiments)
                else:
                    print("Invalid choice or DistilBERT is not available.")

            elif analysis_num == '5':
                data_analysis.list_suitable_variables('Regression')
                dependent_var = input("Enter the dependent variable: ")
                independent_vars = input("Enter the independent variable: ")
                data_analysis.regression_analysis(dependent_var, independent_vars)
            elif analysis_num == '6':
                break

            else:
                print("Invalid input. Please choose a valid option.")
    elif(menu_num=='5'):
        break
    else:
        print("******Error Input******")
        print("Please give the right number, Try angin!")
        continue