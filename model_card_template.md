# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

    - Model Type: Random Forest
    - Version: 1.0
    - Date: 2024-11-07

## Intended Use

    - Primary Intended Uses: 
        The model is intended to predict the likelihood of a individual having a income of above 50K.
    - Primary Intended Users:
        The model is intended to be used by various constituents that include income as an important input to their model. 

## Training Data

    - Source of the Data: 
        The data was sourced from the UCI Machine Learning Repository.
        Link: https://archive.ics.uci.edu/dataset/20/census+income
    - Size of the Dataset: 
        The dataset contains 32561 rows and 15 columns.
    - Data Preprocessing:
        The data was preprocessed by removing missing values and encoding categorical variables.

## Evaluation Data

    - Evaluation Data: 
        The model was evaluated on a test set that was randomly sampled from the original dataset.
    - Size of the Dataset:
        The test set contains 6512 rows and 15 columns.

## Metrics
    
        - The model was evaluated using the following metrics:
            - Precision: 0.7074047447879224
            - Recall: 0.6465177398160316
            - F1 Score: 0.6755921730175077

## Ethical Considerations

    - Potential Biases:
        The model may be biased towards certain groups in the dataset.
    - Fairness:
        The model may not be fair to certain groups in the dataset.
   
## Caveats and Recommendations

    - Caveats:
        The model may not generalize well to other datasets.
    - Recommendations:
        Further evaluation of the model is recommended on a diverse set of data.
