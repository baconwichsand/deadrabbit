import shutil
import os
import datetime

def file_update_backup(filename, backup):
    if os.path.isfile(filename):
        shutil.copyfile(filename, backup)
    os.remove(filename)
