from io import StringIO

import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ReadCSV

""""
    Created by Software Engineer Isa Kulaksiz
    Created time 18.03.2022 / dd.mm.yyyy
"""

"The validation part is selected from the train data set."


def validation(self):
    df = ReadCSV.FileOperations.read_train_file(self)
    df_test = ReadCSV.FileOperations.read_test_file(self)

    ## feature selection
    # print(df.dtypes)
    df_copy_test = df[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission", "Stay"]].copy()
    print(df_copy_test.value_counts())
    # print("TRAIN DATA")
    print(df_copy_test.head(20))
    # 0 ->  gynecology / 1 -> anesthesia / 2-> radiotherapy / 3 -> TB & Chest disease / 4 -> surgery
    # print(df["Department"].value_counts())
    df_copy_test = df_copy_test.replace(['gynecology'], '0')
    df_copy_test = df_copy_test.replace(['anesthesia'], '1')
    df_copy_test = df_copy_test.replace(['radiotherapy'], '2')
    df_copy_test = df_copy_test.replace(['TB & Chest disease'], '3')
    df_copy_test = df_copy_test.replace(['surgery'], '4')
    # print(df_copy_test["Department"].value_counts())

    # 0 -> Moderate / 1 -> Minor / 2 -> Extreme / 3 -> Severity of Illness
    # print(df["Severity of Illness"].value_counts())
    df_copy_test = df_copy_test.replace(['Moderate'], '0')
    df_copy_test = df_copy_test.replace(['Minor'], '1')
    df_copy_test = df_copy_test.replace(['Extreme'], '2')
    # print(df_copy_test["Severity of Illness"].value_counts())

    # 0 -> Trauma / 1 -> Emergency / 2 -> Urgent
    # print(df["Type of Admission"].value_counts())
    df_copy_test = df_copy_test.replace(['Trauma'], '0')
    df_copy_test = df_copy_test.replace(['Emergency'], '1')
    df_copy_test = df_copy_test.replace(['Urgent'], '2')
    # print(df_copy_test["Type of Admission"].value_counts())

    # 0 -> 41-50 / 1 -> 31-40 / 2 -> 51-60 / 3 -> 21-30 / 4 -> 71-80 / 5 -> 61-70
    # / 6 -> 11-20 / 7 -> 81-90 / 8 -> 0-10 / 9 -> 91-100
    # print(df["Age"].value_counts())
    df_copy_test = df_copy_test.replace(['41-50'], '0')
    df_copy_test = df_copy_test.replace(['31-40'], '1')
    df_copy_test = df_copy_test.replace(['51-60'], '2')
    df_copy_test = df_copy_test.replace(['21-30'], '3')
    df_copy_test = df_copy_test.replace(['71-80'], '4')
    df_copy_test = df_copy_test.replace(['61-70'], '5')
    df_copy_test = df_copy_test.replace(['11-20'], '6')
    df_copy_test = df_copy_test.replace(['81-90'], '7')
    df_copy_test = df_copy_test.replace(['0-10'], '8')
    df_copy_test = df_copy_test.replace(['91-100'], '9')
    # print(df_copy_test["Age"].value_counts())

    # 0 -> 21-30 / 1 -> 11-20 / 2 -> 31-40 / 3 -> 51-60 / 4 -> 0-10 / 5 -> 41-50
    # 6 -> 71-80 / 7 -> More than 100 Days /  8 -> 81-90 / 9 -> 91-100 / 10 -> 61-70
    # print(df["Stay"].value_counts())
    df_copy_test = df_copy_test.replace(['21-30'], '0')
    df_copy_test = df_copy_test.replace(['11-20'], '1')
    df_copy_test = df_copy_test.replace(['31-40'], '2')
    df_copy_test = df_copy_test.replace(['51-60'], '3')
    df_copy_test = df_copy_test.replace(['0-10'], '4')
    df_copy_test = df_copy_test.replace(['41-50'], '5')
    df_copy_test = df_copy_test.replace(['71-80'], '6')
    df_copy_test = df_copy_test.replace(['More than 100 Days'], '7')
    df_copy_test = df_copy_test.replace(['81-90'], '8')
    df_copy_test = df_copy_test.replace(['91-100'], '9')
    df_copy_test = df_copy_test.replace(['61-70'], '10')
    # print(df_copy_test["Stay"].value_counts())

    x_train = df_copy_test[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission"]]
    y_train = df_copy_test["Stay"].values

    ## FEATURE EXTRACTION
    options_sol = ['2']
    rslt_df = df_copy_test.loc[df_copy_test['Severity of Illness'].isin(options_sol)]
    print('\nResult Severity of Illness :\n',
          rslt_df)

    options_age = ['4', '5', '7', '9']
    rslt_df_age = df_copy_test.loc[df_copy_test['Age'].isin(options_age)]
    print('\nResult Age :\n',
          rslt_df_age)

    df_feature_ext = df_copy_test.copy()
    common = rslt_df.merge(rslt_df_age, left_index=True, right_index=True, how='inner', suffixes=('', '_y'))
    common.drop(common.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
    print("merged two column : ", common["Stay"])
    print(common.isnull().sum())
    common.loc[common["Hospital_code"].isnull(), "Hospital_code"] = "0"
    common.loc[common["patientid"].isnull(), "patientid"] = "0"
    common.loc[common["Department"].isnull(), "Department"] = "0"
    common.loc[common["Age"].isnull(), "Age"] = "0"
    common.loc[common["Severity of Illness"].isnull(), "Severity of Illness"] = "0"
    common.loc[common["Type of Admission"].isnull(), "Type of Admission"] = "0"
    common.loc[common["Stay"].isnull(), "Stay"] = "0"
    print(common.isnull().sum())

    f = open("new_train.csv", "w")
    print("File has been created!")
    for (i, row) in common.iterrows():
        if common["Hospital_code"][i] == "0" and common["patientid"][i] == "0" and common["Department"][i] == "0" and common["Age"][i] == "0" and common["Severity of Illness"][i] == "0" and common["Type of Admission"][i] == "0" and common["Stay"][i] == "0":
            row["priority"] = "NO"

        else:
            row["priority"] = "YES"
        f.write(str(row["Hospital_code"]) + "," + str(row["patientid"]) + "," + str(row["Department"]) + "," + str(row["Age"]) + "," + str(row["Severity of Illness"]) + "," + str(row["Type of Admission"]) + "," + str(row["Stay"]) + "," + str(row["priority"]) + "\n")
    file = open("new_train.csv", "r")
    df_common = pd.read_csv(file)
    print(df_common.iloc[0:10])
    print(df_common.shape)
    print("null values",df_common.isnull().sum().sum())
    f.close()

    # common.dropna(inplace=True)

    # print("TEST DATA")
    # print(df_test.dtypes)
    ## feature selection
    df_copy_test_data = df_test[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission"]].copy()

    # 0 ->  gynecology / 1 -> anesthesia / 2-> radiotherapy / 3 -> TB & Chest disease / 4 -> surgery
    # print(df_test["Department"].value_counts())
    df_copy_test_data = df_copy_test_data.replace(['gynecology'], '0')
    df_copy_test_data = df_copy_test_data.replace(['anesthesia'], '1')
    df_copy_test_data = df_copy_test_data.replace(['radiotherapy'], '2')
    df_copy_test_data = df_copy_test_data.replace(['TB & Chest disease'], '3')
    df_copy_test_data = df_copy_test_data.replace(['surgery'], '4')
    # print(df_copy_test_data["Department"].value_counts())

    # 0 -> Moderate / 1 -> Minor / 2 -> Extreme / 3 -> Severity of Illness
    # print(df_test["Severity of Illness"].value_counts())
    df_copy_test_data = df_copy_test_data.replace(['Moderate'], '0')
    df_copy_test_data = df_copy_test_data.replace(['Minor'], '1')
    df_copy_test_data = df_copy_test_data.replace(['Extreme'], '2')
    # print(df_copy_test_data["Severity of Illness"].value_counts())

    # 0 -> Trauma / 1 -> Emergency / 2 -> Urgent
    # print(df_test["Type of Admission"].value_counts())
    df_copy_test_data = df_copy_test_data.replace(['Trauma'], '0')
    df_copy_test_data = df_copy_test_data.replace(['Emergency'], '1')
    df_copy_test_data = df_copy_test_data.replace(['Urgent'], '2')
    # print(df_copy_test_data["Type of Admission"].value_counts())

    # 0 -> 41-50 / 1 -> 31-40 / 2 -> 51-60 / 3 -> 21-30 / 4 -> 71-80 / 5 -> 61-70
    # / 6 -> 11-20 / 7 -> 81-90 / 8 -> 0-10 / 9 -> 91-100
    # print(df_test["Age"].value_counts())
    df_copy_test_data = df_copy_test_data.replace(['41-50'], '0')
    df_copy_test_data = df_copy_test_data.replace(['31-40'], '1')
    df_copy_test_data = df_copy_test_data.replace(['51-60'], '2')
    df_copy_test_data = df_copy_test_data.replace(['21-30'], '3')
    df_copy_test_data = df_copy_test_data.replace(['71-80'], '4')
    df_copy_test_data = df_copy_test_data.replace(['61-70'], '5')
    df_copy_test_data = df_copy_test_data.replace(['11-20'], '6')
    df_copy_test_data = df_copy_test_data.replace(['81-90'], '7')
    df_copy_test_data = df_copy_test_data.replace(['0-10'], '8')
    df_copy_test_data = df_copy_test_data.replace(['91-100'], '9')
    # print(df_copy_test_data["Age"].value_counts())

    x_test = df_copy_test_data[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission"]]

    dt = DecisionTreeClassifier(criterion="entropy", random_state=1234, max_depth=4, min_samples_split=4)
    model = dt.fit(x_train, y_train)

    # %25 validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.25,
                                                      shuffle=False)

    y_train = y_train[0:len(x_test):]

    prediction = dt.predict(x_test)
    accuracy = accuracy_score(y_train, prediction)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('\n' + "Confusion Matrix: " + '\n', confusion_matrix(y_train, prediction))
    # print("Report :" + '\n', classification_report(y_train, prediction))

    print(model)

    # validation process
    prediction = dt.predict(x_val)
    print("\r\n_______________________________________________________\r\n")
    # accuracy = accuracy_score(valid['Stay'], valid['prediction'])
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # print('\n' + "Confusion Matrix: " + '\n', confusion_matrix(valid['Stay'], valid['prediction']))
    accuracy = accuracy_score(y_val, prediction)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('\n' + "Confusion Matrix: " + '\n', confusion_matrix(y_val, prediction))
    # print("Report :" + '\n', classification_report(y_val, prediction))
    # Training Dataset
    column_name = ['Hospital_code', 'patientid', 'Department',
                   'Age', 'Severity of Illness', 'Type of Admission']

    data = x_train
    x = np.hstack(x_train)
    y = np.array([50, 200, 1000, 1500, 2000, 2500])

    plt.bar(x, y)
    # plt.show()

    return df
