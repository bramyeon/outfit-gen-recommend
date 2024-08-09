[AI506-PROJECT-README]

    Design of Predictive Algorithms for Outfit Generation and Recommendation
                     Spring 2023 AI506 Term Project Readme

          Bryan Nathanael Wijaya, 20200735 (bryannwijaya@kaist.ac.kr)
               Quang Minh Nguyen, 20200854 (qm.nguyen@kaist.ac.kr)
                    School of Electrical Engineering, KAIST
    INSTRUCTOR: Prof. Kijung Shin, Kim Jaechul Graduate School of AI, KAIST

================================================================================

1. HELPING INDIVIDUALS 
(including friends, classmates, lab TA, course staff members, etc.)
+--------------+---------------------------------------------------------------+
|     NAME     |                         NATURE OF HELP                        |
+==============+===============================================================+
|              |							       |
+--------------+---------------------------------------------------------------+


2. HOW TO RUN CODES

Our code assumes that the following files are present in "./dataset/":

- itemset_item_test_query.csv 
- itemset_item_training_csv
- itemset_item_valid_answer.csv
- itemset_item_valid_query.csv
- user_item.csv 
- user_itemset_test_query.csv
- user_itemset_training.csv
- user_itemset_valid_answer.csv
- user_itemset_valid_query.csv

You might need to create a writable directory "./result/".

Furthermore, the codes should be run in a conda environment containing
the following packages:

numpy, matplotlib, scipy, sklearn, tqdm, os, collections, time,
itertools, random, re, csv, ipykernel


2.1. Task 1: Outfit Recommendation

Run the following command in the command line:

python sfi.py dataset/user_itemset_training.csv dataset/user_itemset_test_query.csv

This command should produce "./result/user_itemset_test_prediction.csv", a file
containing our predictions.

If you want to reproduce our experiments 1 and 3 for task 1
and get visualizations of the results, open task1_fi.ipynb and
run all. This will take around 5 minutes, due to our grid search.

2.2. Task 2: Outfit Generation

Open project_task2.ipynb. Then, run Steps 1, 2, 6, and 8 in the notebook 
(in Step 6, running only the first 4 code blocks is enough). These steps 
should produce "./result/itemset_item_test_prediction.csv", a file 
containing our predictions.


3. COMMENTS FOR ASSIGNMENT GRADING


4. EVALUATION OF ASSIGNMENT


5. IDEAS

