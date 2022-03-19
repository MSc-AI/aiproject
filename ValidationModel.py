import numpy as np
import pandas as pd
from self import self
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import ReadCSV

""""
    Created by Software Engineer Isa Kulaksiz
    Created time 18.03.2022 / dd.mm.yyyy
"""

"The validation part is selected from the train data set."

def validation(self):
    df = ReadCSV.FileOperations.read_train_file(self)
    df_test = ReadCSV.FileOperations.read_test_file(self)

    print(df.dtypes)
    df_copy_test = df.copy()

    # 0 ->  gynecology / 1 -> anesthesia / 2-> radiotherapy / 3 -> TB & Chest disease / 4 -> surgery
    print(df["Department"].value_counts())
    df_copy_test = df_copy_test.replace(['gynecology'], '0')
    df_copy_test = df_copy_test.replace(['anesthesia'], '1')
    df_copy_test = df_copy_test.replace(['radiotherapy'], '2')
    df_copy_test = df_copy_test.replace(['TB & Chest disease'], '3')
    df_copy_test = df_copy_test.replace(['surgery'], '4')
    print(df_copy_test["Department"].value_counts())

    # 0 -> Moderate / 1 -> Minor / 2 -> Extreme / 3 -> Severity of Illness
    print(df["Severity of Illness"].value_counts())
    df_copy_test = df_copy_test.replace(['Moderate'], '0')
    df_copy_test = df_copy_test.replace(['Minor'], '1')
    df_copy_test = df_copy_test.replace(['Extreme'], '2')
    print(df_copy_test["Severity of Illness"].value_counts())

    # 0 -> Trauma / 1 -> Emergency / 2 -> Urgent
    print(df["Type of Admission"].value_counts())
    df_copy_test = df_copy_test.replace(['Trauma'], '0')
    df_copy_test = df_copy_test.replace(['Emergency'], '1')
    df_copy_test = df_copy_test.replace(['Urgent'], '2')
    print(df["Type of Admission"].value_counts())




    x_train = df_copy_test[["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission"]]
    y_train = df_copy_test[["Stay"]]




    dt = DecisionTreeClassifier(criterion="entropy", random_state=1234, max_depth=4, min_samples_split=4)
    model = dt.fit(x_train, y_train)
    print(model)


    return df
