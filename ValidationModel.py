import numpy as np
from self import self
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import matplotlib.pyplot as plt
import ReadCSV

""""
    Created by Software Engineer Isa Kulaksiz
    Created time 18.03.2022 / dd.mm.yyyy
"""

"The validation part is selected from the train data set."

df = ReadCSV.FileOperations.read_train_file(self)
df_test = ReadCSV.FileOperations.read_test_file(self)


def feature_selection():
    # print(df.dtypes)
    df_train = df[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission", "Stay"]].copy()
    print(df_train.value_counts())
    # print("TRAIN DATA")
    print(df_train.head(20))
    # 0 ->  gynecology / 1 -> anesthesia / 2-> radiotherapy / 3 -> TB & Chest disease / 4 -> surgery
    # print(df["Department"].value_counts())
    df_train = df_train.replace(['gynecology'], '0')
    df_train = df_train.replace(['anesthesia'], '1')
    df_train = df_train.replace(['radiotherapy'], '2')
    df_train = df_train.replace(['TB & Chest disease'], '3')
    df_train = df_train.replace(['surgery'], '4')
    # print(df_train["Department"].value_counts())

    # 0 -> Moderate / 1 -> Minor / 2 -> Extreme / 3 -> Severity of Illness
    # print(df["Severity of Illness"].value_counts())
    df_train = df_train.replace(['Moderate'], '0')
    df_train = df_train.replace(['Minor'], '1')
    df_train = df_train.replace(['Extreme'], '2')
    # print(df_train["Severity of Illness"].value_counts())

    # 0 -> Trauma / 1 -> Emergency / 2 -> Urgent
    # print(df["Type of Admission"].value_counts())
    df_train = df_train.replace(['Trauma'], '0')
    df_train = df_train.replace(['Emergency'], '1')
    df_train = df_train.replace(['Urgent'], '2')
    # print(df_train["Type of Admission"].value_counts())

    # 0 -> 41-50 / 1 -> 31-40 / 2 -> 51-60 / 3 -> 21-30 / 4 -> 71-80 / 5 -> 61-70
    # / 6 -> 11-20 / 7 -> 81-90 / 8 -> 0-10 / 9 -> 91-100
    # print(df["Age"].value_counts())
    df_train = df_train.replace(['41-50'], '0')
    df_train = df_train.replace(['31-40'], '1')
    df_train = df_train.replace(['51-60'], '2')
    df_train = df_train.replace(['21-30'], '3')
    df_train = df_train.replace(['71-80'], '4')
    df_train = df_train.replace(['61-70'], '5')
    df_train = df_train.replace(['11-20'], '6')
    df_train = df_train.replace(['81-90'], '7')
    df_train = df_train.replace(['0-10'], '8')
    df_train = df_train.replace(['91-100'], '9')
    # print(df_train["Age"].value_counts())

    # 0 -> 21-30 / 1 -> 11-20 / 2 -> 31-40 / 3 -> 51-60 / 4 -> 0-10 / 5 -> 41-50
    # 6 -> 71-80 / 7 -> More than 100 Days /  8 -> 81-90 / 9 -> 91-100 / 10 -> 61-70
    # print(df["Stay"].value_counts())
    df_train = df_train.replace(['21-30'], '0')
    df_train = df_train.replace(['11-20'], '1')
    df_train = df_train.replace(['31-40'], '2')
    df_train = df_train.replace(['51-60'], '3')
    df_train = df_train.replace(['0-10'], '4')
    df_train = df_train.replace(['41-50'], '5')
    df_train = df_train.replace(['71-80'], '6')
    df_train = df_train.replace(['More than 100 Days'], '7')
    df_train = df_train.replace(['81-90'], '8')
    df_train = df_train.replace(['91-100'], '9')
    df_train = df_train.replace(['61-70'], '10')
    # print(df_train["Stay"].value_counts())
    return df_train


def feature_extraction():
    df_copy_train = feature_selection()
    options_sol = ['2']
    rslt_df = df_copy_train.loc[df_copy_train['Severity of Illness'].isin(options_sol)]
    print('\nResult Severity of Illness :\n',
          rslt_df)

    options_age = ['4', '5', '7', '9']
    rslt_df_age = df_copy_train.loc[df_copy_train['Age'].isin(options_age)]
    print('\nResult Age :\n',
          rslt_df_age)

    df_feature_ext = df_copy_train.copy()
    print("rslt_df size:" + str(rslt_df.shape))
    common = rslt_df.merge(rslt_df_age, left_index=True, right_index=True, how='outer', suffixes=('', '_drop'))
    common.drop(common.filter(regex='_y$').columns.tolist(), axis=1, inplace=False)
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

    f = open("train_join.csv", "w")
    f.write("Hospital_code,patientid,Department,Age,Severity of Illness,Type of Admission,priority,Stay\n")
    print("File has been created!")
    for (i, row) in common.iterrows():
        if common["Hospital_code"][i] == "0" and common["patientid"][i] == "0" and common["Department"][i] == "0" and \
                common["Age"][i] == "0" and common["Severity of Illness"][i] == "0" and common["Type of Admission"][
            i] == "0" and common["Stay"][i] == "0":
            row["Hospital_code"] = df_copy_train["Hospital_code"][i]
            row["patientid"] = df_copy_train["patientid"][i]
            row["Department"] = df_copy_train["Department"][i]
            row["Age"] = df_copy_train["Age"][i]
            row["Severity of Illness"] = df_copy_train["Severity of Illness"][i]
            row["Type of Admission"] = df_copy_train["Type of Admission"][i]
            row["Stay"] = df_copy_train["Stay"][i]

            # row["priority"] = "NO"
            row["priority"] = "0"

        else:
            # row["priority"] = "YES"
            row["priority"] = "1"

        f.write(str(row["Hospital_code"]) + "," + str(row["patientid"]) + "," + str(row["Department"]) + "," + str(
            row["Age"]) + "," + str(row["Severity of Illness"]) + "," + str(row["Type of Admission"]) + "," +
                str(row["priority"]) + "," + str(row["Stay"]) + "\n")
    file = open("train_join.csv", "r")
    df_common = pd.read_csv(file)

    print(df_common.iloc[0:10])
    print(df_common.shape)
    print("null values", df_common.isnull().sum().sum())
    f.close()
    return df_common


def feature_selection_test():
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
    return df_copy_test_data


def validation(self):
    df_train = feature_selection()
    df_test = feature_selection_test()

    x_test = df_test[["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission"]]
    x_train = df_train[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission"]]
    y_train = df_train[["Stay"]]
    y_test = df_train[["Stay"]]

    df_common = feature_extraction()
    df_copy_test_data = feature_selection_test()

    ## --------------TEST DATA-----------------
    dt = DecisionTreeClassifier()
    y_test = y_test[0:len(x_test):]
    dt.fit(x_test, y_test)
    prediction_ = dt.predict(x_test)
    accuracy = accuracy_score(y_test, prediction_)
    print("\r\n____________________________________________________\r\n")
    print("TEST DATA")
    print("\r\n____________________________________________________\r\n")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('\n' + "Confusion Matrix: " + '\n', confusion_matrix(y_test, prediction_))
    # print("Report :" + '\n', classification_report(y_train_feat, prediction))
    print("\r\n______________________END______________________________\r\n")

    ## --------------TRAIN DATA-----------------
    dt.fit(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      train_size=0.70, shuffle=False)

    # y_train = y_train[0:len(x_test):]
    prediction = dt.predict(x_train)
    accuracy = accuracy_score(y_train, prediction)
    print("\r\n____________________________________________________\r\n")
    print("TRAIN DATA")
    print("\r\n____________________________________________________\r\n")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('\n' + "Confusion Matrix: " + '\n', confusion_matrix(y_train, prediction))
    # print("Report :" + '\n', classification_report(y_train, prediction))
    print("\r\n______________________END______________________________\r\n")


    ## --------------VALIDATION DATA-----------------
    prediction = dt.predict(x_val)
    accuracy = accuracy_score(y_val, prediction)
    print("\r\n____________________________________________________\r\n")
    print("TRAIN VALIDATION DATA")
    print("\r\n____________________________________________________\r\n")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('\n' + "Confusion Matrix: " + '\n', confusion_matrix(y_val, prediction))
    # print("Report :" + '\n', classification_report(y_val, prediction))
    print("\r\n____________________END________________________________\r\n")

    ## FEATURE EXTRACTION ALGORTIHM
    options_sol = ['2']
    rslt_df_test = df_copy_test_data.loc[df_copy_test_data['Severity of Illness'].isin(options_sol)]
    # print('\nResult Severity of Illness :\n', rslt_df_test)

    options_age = ['4', '5', '7', '9']
    rslt_df_test_age = df_copy_test_data.loc[df_train['Age'].isin(options_age)]
    # print('\nResult Age :\n', rslt_df_test_age)

    common = rslt_df_test.merge(rslt_df_test_age, left_index=True, right_index=True, how='outer',
                                suffixes=('', '_drop'))
    common.drop(common.filter(regex='_y$').columns.tolist(), axis=1, inplace=False)

    common.loc[common["Hospital_code"].isnull(), "Hospital_code"] = "0"
    common.loc[common["patientid"].isnull(), "patientid"] = "0"
    common.loc[common["Department"].isnull(), "Department"] = "0"
    common.loc[common["Age"].isnull(), "Age"] = "0"
    common.loc[common["Severity of Illness"].isnull(), "Severity of Illness"] = "0"
    common.loc[common["Type of Admission"].isnull(), "Type of Admission"] = "0"
    # print(common.isnull().sum())

    f = open("test_join.csv", "w")
    f.write("Hospital_code,patientid,Department,Age,Severity of Illness,Type of Admission,priority\n")
    for (i, row) in common.iterrows():
        if common["Hospital_code"][i] == "0" and common["patientid"][i] == "0" and common["Department"][i] == "0" and \
                common["Age"][i] == "0" and common["Severity of Illness"][i] == "0" and common["Type of Admission"][
            i] == "0":
            row["Hospital_code"] = df_train["Hospital_code"][i]
            row["patientid"] = df_train["patientid"][i]
            row["Department"] = df_train["Department"][i]
            row["Age"] = df_train["Age"][i]
            row["Severity of Illness"] = df_train["Severity of Illness"][i]
            row["Type of Admission"] = df_train["Type of Admission"][i]

            # row["priority"] = "NO"
            row["priority"] = "0"

        else:
            # row["priority"] = "YES"
            row["priority"] = "1"

        f.write(str(row["Hospital_code"]) + "," + str(row["patientid"]) + "," + str(row["Department"]) + "," + str(
            row["Age"]) + "," + str(row["Severity of Illness"]) + "," + str(row["Type of Admission"]) + ","
                + str(row["priority"]) + "\n")
    file_test = open("test_join.csv", "r")
    df_test_common = pd.read_csv(file_test)
    x_train_feat = df_common[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission", "priority"]]
    print("x_train: ", x_train_feat.shape)
    y_train_feat = df_common[["Stay"]]
    print("y_train: ", y_train_feat.shape)
    x_test_feat = df_test_common[
        ["Hospital_code", "patientid", "Department", "Age", "Severity of Illness", "Type of Admission", "priority"]]

    ## Feture Extraction with test data
    dt = DecisionTreeClassifier()
    y_train_feat = y_train_feat[0:len(x_test_feat):]
    model = dt.fit(x_test_feat, y_train_feat)
    print(model)

    """# %25 validation data
    x_train_feat, x_val_test, y_train_feat, y_val_test = train_test_split(x_train_feat, y_train_feat,
                                                                          train_size=0.70,
                                                                          shuffle=False)"""

    prediction_ = dt.predict(x_test_feat)
    accuracy = accuracy_score(y_train_feat, prediction_)
    print("\r\n____________________________________________________\r\n")
    print("FEATURE EXTRACTION ALGORTIHM")
    print("\r\n____________________________________________________\r\n")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('\n' + "Confusion Matrix: " + '\n', confusion_matrix(y_train_feat, prediction_))
    # print("Report :" + '\n', classification_report(y_train_feat, prediction))
    print("\r\n___________________END_________________________________\r\n")

    # Training Dataset
    column_name = ['Hospital_code', 'patientid', 'Department',
                   'Age', 'Severity of Illness', 'Type of Admission']

    x = np.hstack(x_train)
    y = np.array([50, 200, 1000, 1500, 2000, 2500])

    plt.bar(x, y)
    # plt.show()

    return df




