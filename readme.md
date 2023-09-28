## About this Project: 


### Test Model to Show Personal Loan & Credit Pre-Approval Predictive Model Within Banking  

## Key Variables Utilized within Analysis 
* Income - Current Annual Income  
* CCAvg - Monthly Credit Utilization on Credit Card  
* Mortgage - Existing Mortgage Balance else 0  

## Classification Done Through Naive-Bayes & Decision Tree Classifiers
```txt 
Classification Result - Naive Bayes

Confusion Matrix:
 [[811  84]
 [ 27  78]]

Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.91      0.94       895
           1       0.48      0.74      0.58       105

    accuracy                           0.89      1000
   macro avg       0.72      0.82      0.76      1000
weighted avg       0.92      0.89      0.90      1000


Accuracy Score: 0.889

Classification Result - Decision Tree
Confusion Matrix:
 [[888   7]
 [ 25  80]]

Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.99      0.98       895
           1       0.92      0.76      0.83       105

    accuracy                           0.97      1000
   macro avg       0.95      0.88      0.91      1000
weighted avg       0.97      0.97      0.97      1000


Accuracy Score: 0.968
```  
## Naive Bayes Accuracy = 88.90%, Decision Tree = 96.80%