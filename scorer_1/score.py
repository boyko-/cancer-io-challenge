import glob
import os
import pandas as pd
import numpy as np

unzipped_path = './unzipped/'

scoring_type = "test" # can also be "validation"

#===========================
error_default_value = 100000 # default L1 score for erroneous solutions
require_exactly_one = False # how strictly vector sum = 1 needs to be enforced


def convert_to_dict(df):
    # Converts the df to a dict with gene as the key and value being a list of length 5
    global expected_len
    global require_exactly_one
    all_rows = df.values.tolist()

    if len(all_rows) < expected_len:
        return {}, False, "Row count insufficient: " + str(len(all_rows)) 
    else:
        this_dict = {}
        for k, row in enumerate(all_rows[-expected_len:]): 
            # for each row consider only the last 6 columns
            row = row[-6:] 

            # if only 5 columns are found, that indicates that gene name is missing
            if len(row) == 5:
                # in this case, add the gene name, based on row number
                row = [actual_genes[k]] + row

            vector_sum = np.sum([float(a) for a in row[1:]])
            if require_exactly_one:
                if abs(vector_sum - 1) > 0.0001:
                    # raise error if tolerance exceeded
                    return this_dict, False, "Vector sum = " + str(vector_sum) + " | Exceeded the tolerance for sum = 1"
                else:
                    # else let it pass through
                    final_vector = [float(a) for a in row[1:]]
            else:
                # force normalize by dividing by vector sum
                final_vector = [float(a)/vector_sum for a in row[-5:]]

            this_dict[row[0]] = final_vector

        return this_dict, True, "OK"


def score_dicts(pred_dict_this, actual_dict):
    collect_scores = []

    print("pred_dict_this:", pred_dict_this)
    for pred_key in pred_dict_this:
        # fetch the submitted vector for this gene
        pred_vector = np.array(pred_dict_this[pred_key])
        
        print("pred_vector sum:", np.sum(pred_vector))
        if len(pred_vector) == 5:
            # fetch the ground truth vector for this gene
            actual_vector = np.array(actual_dict[pred_key])
            # calculate l1 loss
            l1_loss = np.sum(np.abs(actual_vector - pred_vector))
            collect_scores.append(l1_loss)
        else:
            raise Exception("Prediction vector length is incorrect:" + str(len(pred_vector)) )

    global expected_len
    if len(collect_scores) == expected_len:
        # calculate the mean only if the number of individual l1 scores matches the expected count
        return np.mean(collect_scores)
    else:
        print("Wrong length for collect_scores:", collect_scores)
        raise Exception("Total L1 score count found unsufficient to take the mean:" + str(len(collect_scores)))


#=======================================================================
df_actual = pd.read_csv("./ground_truth_" + scoring_type + ".csv", header = None)
expected_len = df_actual.shape[0] - 1
actual_dict, _, _ = convert_to_dict(df_actual)
actual_genes = list(actual_dict.keys())
df_actual = None
#===============================

idx = 1
folders = []
scores = []
issues = []
list_ = glob.glob(unzipped_path + "*" + os.path.sep)
list_ = sorted(list_, key=str.lower)


for sub_id_this, submission_folder in enumerate(sorted(list_)):
    print("*----------------------------------------------------------------------------------------*")
    print(sub_id_this, submission_folder)

    try:
    #if True:
        subfolders = [f.path for f in os.scandir("./" + submission_folder)]
        print("Subfolders:", subfolders)
        solution_path = None
        if os.path.exists(submission_folder + "solution"):
            # look for solution folder in root
            solution_path = submission_folder + "solution"
        else:
            for subfolder in subfolders:
                # look for solution folder in the sub-folders
                if os.path.exists(subfolder + "/solution"):
                    solution_path = subfolder + "/solution"
                    break

        if solution_path is None:
            l1_loss = error_default_value
            status = "/solution/" + scoring_type + "_output.csv not found"
        else:
            df = pd.read_csv(solution_path + "/" + scoring_type + "_output.csv", engine='python', header = None).dropna(axis=1)

            pred_dict, conversion_status, status = convert_to_dict(df)

            for pred_key in pred_dict.keys():
                if pred_key not in actual_genes:
                    raise Exception('Unexpected gene name found at 6th column from the right. Should be exactly one of these: ' + ", ".join(actual_genes))

            if conversion_status:
                print(df)
                print(".......")
                print(pred_dict)
                df = None
                l1_loss = score_dicts(pred_dict, actual_dict)
                print("L1 Loss:", l1_loss)
            else:
                l1_loss = error_default_value

    except BaseException as e:
        print(idx, submission_folder)
        print("Error")
        print(e)
        l1_loss = error_default_value
        status = str(e)
    
    pred_dict = None
    df = None

    issues.append(("Error: " if status != "OK" else "") + status)

    idx += 1
    folders.append(submission_folder.split("/")[2])
    scores.append(l1_loss)



print("require_exactly_one:", require_exactly_one)
print("------------")
print("expected_len:", expected_len)
print("actual_genes:", actual_genes)
print("-------------------------------------------------------------------------------")
print("Submission Ranking:")
print("RANK | Submission ID | Mean L1 Loss | Status")
last_value = 0
rank = 1
index = 0


same_scores = {}
for rank_id, k in enumerate(np.argsort(np.array(scores))):
    print("------")
    if scores[k] != last_value:
        if  rank_id > 0 and len(same_scores[last_value]) == 1:
            del same_scores[last_value]

        if scores[k] == error_default_value:
            print("................................")
        rank = rank_id + 1
        
        
    if scores[k] not in same_scores:
        same_scores[scores[k]] = []   
    same_scores[scores[k]].append(folders[k])

    index += 1
    print(rank, "|", folders[k], "|", scores[k], "|", issues[k][:115].replace("././unzipped/", ""))   
    
    last_value = scores[k]

print("========================================")
print("Same scoring submissions:", same_scores)
print("scoring_type:", scoring_type)