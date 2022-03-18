import io
import os


class FileOperations:
    def read_file(self):

        global file
        desktop_path = os.path.join(os.path.join(os.environ['HOME']), 'Desktop')

        arr = ""
        file_path_out = f"{desktop_path}//healthcare/test_data.csv"
        try:
            if not os.path.exists(file_path_out):
                open(file_path_out, "w").close()
            else:
                file = io.open(file_path_out, "r+", encoding="UTF-8")
        except IOError:
            print("test_data.csv not found!")
        for i in file.readlines():
            arr += i
        print(arr)
        return arr
