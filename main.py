""""
    Created by Software Engineer Isa Kulaksiz
    Created time 18.03.2022 / dd.mm.yyyy
"""
from TestData import FileOperations

if __name__ == '__main__':
    temp_data = FileOperations().read_file()
    if temp_data.read_file() == "":
        print("test_data.csv is empty !")

