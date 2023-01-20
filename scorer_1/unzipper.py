import zipfile

import glob
# folder where the downloaded submissions are placed
submission_folder_name = "submission_zipped"

# folder where unzipped folders will end up
unzipped_folder_name = "unzipped"

for k, filepath in enumerate(sorted(glob.iglob('./' + submission_folder_name + '/*.zip'))):
    print(filepath)
    new_path = "./" + unzipped_folder_name + "/" + filepath.split("-")[1] 
    print(k + 1, new_path)
    print("---------")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(new_path)

print("===========================")
print("Done")