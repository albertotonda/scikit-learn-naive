# Naive 10-fold cross-validation over scikit-learn models

This is essentially a basic script to run a 10-fold cross-validation over all classification or regression models found in scikit-learn. An ensemble evaluation can be useful to identify the most promising algorithms (on which later focus for hyperparameter optimization) or to understand whether there is any hope of finding good results for the data set (for example, if no algorithm performs above a user-defined threshold, very likely it will be difficult to obtain anything).

## TODO
1. it would be nice to have a refactoring, with subdirectories `/src` and `/data`, plus a `/results` that is not part of the version control.
2. Global plot with comparison of the 10 most performing algorithms, as a histogram or boxplot.
3. Subfolders with all the details for each algorithm, instead of a single very confusing folder.
4. Sanitize the names of the input variables, to avoid a mess with file naming?
5. Keep track of `y = f(X)`, with all the names of the variables, either in the log, or in a separate file.

### naive-regression
1. Keep track of the performance of each algorithm on each fold.
2. Replace the current plots with the scikit-learn library plots (maybe using seaborn for prettier graphics).

### naive-classification
1. Add all other metrics (ROC AUC, F1, MCC, etc.).
2. Store performance of each algorithm on each fold.