The "test.csv" file contains the test data for evaluation.
The file contains 486 tweets labelled with the concerns towards vaccines. The file is formatted in the standard CSV format, and can be viewed using spreadsheet software (e.g. Libre Office, Microsoft Excel). 
To read the file in Python, you may use the in-built "csv" library, or the "pandas" library.


There are 2 columns in the file:
 - ID of the tweet in a string format, appended with a "t" (to make it easier to work with on spreadsheet softwares).
 - The tweet text


To submit your runs, please prepare a similar CSV file with the ID column, and a column for the predictions. Each of your predictions will be a space-separated list of classes. Thus the file will look like:
"id", "preds"
xxxxxx1t, none
xxxxxx2t, rushed ineffective
xxxxxx2t, side-effect ineffective mandatory
...

NOTE: Make sure the IDs are exactly the same as in the test.csv file.

