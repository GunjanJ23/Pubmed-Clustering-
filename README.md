# Pubmed-Clustering-

Clustering project on Pubmed Data

## Aim

These PMIDs are the partial results of PubMed searches for different diseases which were subsequently combined and shuffled.
We would like you to retrieve the abstracts for those PMIDs and cluster them into groups that would ideally match the search query results.


## Data

1. pmids_gold_set_labeled.txt contains a 'gold set' of PMIDs labeled with the search terms used to retrieve them
2. pmids_gold_set_unlabeled.txt contains the same gold set of PMIDS, combined and shuffled, but with labels removed.
3. pmids_test_set_unlabeled.txt contains a separate 'test set' of PMIDs with no search term labels.

## Requirements

1. Python 3
2. Matplot Library 


## Run 

1. Enter the address of your folder which contains the code file and the datasets in your terminal.

    For example: 
    
    ```
    C:\Users\GunjanJ\Desktop\pubmed_clustering_assignment\instructions
    ```
 2. Now, enter the address of the python.exe file, the second argument is your code file and the last argument is your training file.
 
    ```
     C:\Users\GunjanJ\AppData\Local\Programs\Python\Python35\python.exe code.py pmids_gold_set_labeled
    .txt
    ```
 3. After running, the program will output the necessary plots along with the pmids_test_set_labeled.txt file which contains the id along 
    with the cluster labels.
    
    
