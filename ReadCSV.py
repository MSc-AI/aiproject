import io
import os


class FileOperations:
    def read_test_file(self):

        global file
        desktop_path = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

        arrTest = ""
        file_path_out = f"{desktop_path}//healthcare/test_data.csv"
        try:
            if not os.path.exists(file_path_out):
                open(file_path_out, "w").close()
            else:
                file = io.open(file_path_out, "r+", encoding="UTF-8")
        except IOError:
            print("test_data.csv not found!")
        for i in file.readlines():
            arrTest += i
        return arrTest
    def read_train_file(self):

        global file
        desktop_path = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

        arrTrain = ""
        file_path_out = f"{desktop_path}//healthcare/train_data.csv"
        try:
            if not os.path.exists(file_path_out):
                open(file_path_out, "w").close()
            else:
                file = io.open(file_path_out, "r+", encoding="UTF-8")
        except IOError:
            print("test_data.csv not found!")
        for i in file.readlines():
            arrTrain += i
        print(arrTrain)
        return arrTrain
