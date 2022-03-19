import io
import os
import pandas as pd

class FileOperations:
    def read_test_file(self):

        global file, df
        desktop_path = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

        arrTest = ""
        file_path_out = f"{desktop_path}//healthcare/test_data.csv"
        try:
            if not os.path.exists(file_path_out):
                open(file_path_out, "w").close()
            else:
                df = pd.read_csv(file_path_out)
        except IOError:
            print("test_data.csv not found!")
        only_show_duplicated = df[df.duplicated()]
        print("_________only_show_duplicated__________")
        print(only_show_duplicated)
        print("___________________")
        df_copy = df.copy()
        print("________exclude_missing_values___________")
        exclude_missing_values = df.describe(include='all')
        print(exclude_missing_values)
        print("________calculate_sumof_each_nan_column___________")
        calculate_sumof_each_nan_column = df_copy.isnull().sum()
        print(calculate_sumof_each_nan_column)
        print("________data_math_feat___________")
        data_math_feat = df_copy.describe()
        print(data_math_feat)
    def read_train_file(self):

        global file
        desktop_path = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

        arrTrain = ""
        file_path_out = f"{desktop_path}//healthcare/train_data.csv"
        try:
            if not os.path.exists(file_path_out):
                open(file_path_out, "w").close()
            else:
                df = pd.read_csv(file_path_out)
        except IOError:
            print("test_data.csv not found!")

