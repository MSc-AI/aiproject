""""
    Created by Software Engineer Isa Kulaksiz
    Created time 18.03.2022 / dd.mm.yyyy
"""
from ReadCSV import FileOperations

if __name__ == '__main__':
    print("************TEST DATA****************")
    temp_data = FileOperations().read_test_file()
    print("************TRAIN DATA***************")
    train_data = FileOperations().read_train_file()
    if temp_data == "":
        print("test_data.csv is empty !")
    if train_data == "":
        print("train_data.csv is empty!")

