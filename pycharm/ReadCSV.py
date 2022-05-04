import os
import pandas as pd
""""
    Created by Software Engineer Isa Kulaksiz
    Created time 18.03.2022 / dd.mm.yyyy
"""
class FileOperations:
    def read_test_file(self):

        global file, df
        desktop_path = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

        file_path_out = f"{desktop_path}//healthcare/test_data.csv"
        df = pd.read_csv(file_path_out)

        # Data Manipulation
        only_show_duplicated = df[df.duplicated()]
        # print("_________only_show_duplicated__________")
        # print(only_show_duplicated)
        # print("___________________")
        df_test_copy = df.copy()
        # print("________exclude_missing_values___________")
        exclude_missing_values = df.describe(include='all')
        # print(exclude_missing_values)
        # print("________calculate_sumof_each_nan_column___________")
        calculate_sumof_each_nan_column = df_test_copy.isnull().sum()
        # print(calculate_sumof_each_nan_column)
        # print("________data_math_feat___________")
        data_math_feat = df_test_copy.describe()
        # print(data_math_feat)

        # fill in miss values
        bed_grade_mean = df_test_copy["Bed Grade"].mean()
        # print("Bed Grade mean: ", bed_grade_mean)
        city_code_patient_mean = df_test_copy["City_Code_Patient"].mean()
        # print("City_Code_Patient mean: ", city_code_patient_mean)

        # fill nan bed grades
        df_test_copy.loc[df_test_copy["Bed Grade"].isnull(), "Bed Grade"] = bed_grade_mean

        df_test_copy.loc[df_test_copy["City_Code_Patient"].isnull(), "City_Code_Patient"] = city_code_patient_mean
        # print("________Each Column isNotNull___________")
        calculate_sumof_each_isnan_column = df_test_copy.isnull().sum()
        # print(calculate_sumof_each_isnan_column)
        # print(df_test_copy)
        return df_test_copy

    def read_train_file(self):
        global file, df_train
        desktop_path = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

        file_path_out = f"{desktop_path}//healthcare/train_data.csv"

        df_train = pd.read_csv(file_path_out, encoding= 'utf-8')
        # Data Manipulation
        only_show_duplicated = df_train[df_train.duplicated()]
        # print("_________only_show_duplicated__________")
        # print(only_show_duplicated)
        # print("___________________")
        df_copy = df_train.copy()
        # print("________exclude_missing_values___________")
        exclude_missing_values = df_train.describe(include='all')
        # print(exclude_missing_values)
        # print("________calculate_sumof_each_nan_column___________")
        calculate_sumof_each_nan_column = df_copy.isnull().sum()
        # print(calculate_sumof_each_nan_column)
        # print("________data_math_feat___________")
        data_math_feat = df_copy.describe()
        # print(data_math_feat)

        # fill in miss values
        bed_grade_mean = df_copy["Bed Grade"].mean()
        # print("Bed Grade mean: ", bed_grade_mean)
        city_code_patient_mean = df_copy["City_Code_Patient"].mean()
        # print("City_Code_Patient mean: ", city_code_patient_mean)

        # fill nan bed grades
        df_copy.loc[df_copy["Bed Grade"].isnull(), "Bed Grade"] = bed_grade_mean

        df_copy.loc[df_copy["City_Code_Patient"].isnull(), "City_Code_Patient"] = city_code_patient_mean
        # print("________Each Column isNotNull___________")
        calculate_sumof_each_isnan_column = df_copy.isnull().sum()
        # print(calculate_sumof_each_isnan_column)
        return df_copy


