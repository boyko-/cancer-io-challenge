#### Execution Instructions

- The scorer uses `numpy, pandas and zipfile` libraries. Please ensure these are included in your environment. 

- Note that dummy ground truth files are placed in the root of the submission:
  - *ground_truth_validation.csv* (dummy version, for scoring in validation scoring mode)
  - *ground_truth_test.csv* (dummy version, for scoring in final scoring mode)

- If you are a contestant who is testing  your solution, please submit your solution first to Topcoder, and then download your submission back by clicking on your submission ID in [Online Review](https://software.topcoder.com/review/actions/ViewProjectDetails?pid=30324343).

- Note that submissions should be downloaded from Online Review (OR platform) compulsorily, so that the zip files have the default Topcoder generated zip filename, so that the Submission ID can be extracted from the filename, as well as you closely simulate the download and review process.

- Place all downloaded .zip format submissions (should be downloaded from OR so that their filenames are in the correct format) in the `submission_zipped` folder (or whatever folder is set for `submission_folder_name` in unzipper.py). Ensure that no other unnecessary zip files are placed in this folder. 

- Delete any pre-existing files from the 'unzipped' folder. 

- After the submission zip files are placed in the `submission_zipped` folder, run:
`python unzipper.py`

- The command above will extract each submission and place it in the `unzipped` folder, with the extracted Submission ID being the folder name.

- In the file `score.py` ensure `scoring_type = "validation"` for checkpoint/validation scoring, and `scoring_type = "test"` for the final test set based scoring.

- In `score.py` leave `require_exactly_one = False` as is (to force normalize to sum = 1).

- To score the submissions, run:
`python score.py`

- The script will interate through each extracted folder in the `unzipped` folder, and finally render a ranked list for each corresponding Submission ID.