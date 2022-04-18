""""
    Created by Software Engineer Isa Kulaksiz
    Created time 18.03.2022 / dd.mm.yyyy
"""
import self as self

import ValidationModel
from ReadCSV import FileOperations

if __name__ == '__main__':
    # print("************TEST DATA****************")
    temp_data = FileOperations().read_test_file()
    # print("************TRAIN DATA***************")
    train_data = FileOperations().read_train_file()
    # print("\r\n\n_____________________________\n")
    test_manipulated_data = ValidationModel.validation(self)