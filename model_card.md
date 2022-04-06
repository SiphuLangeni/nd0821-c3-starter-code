# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This Random Forest Classifier model was created by Siphu Langeni.

## Intended Use
This model performs a prediction task to determine whether a person makes over 50K a year.

## Training Data
The data is a publicly available dataset found at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income). After data cleaning, there are a total of 32,561 records with 14 features. The model was trained on 80% of the data.

## Evaluation Data
The model was evaluated on a hold-out set of 20% of the records in the original dataset.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

| Metric   |      Value     
|----------|:-------------:|
| precision | 0.75 |
| recall | 0.64 |
| f-beta | 0.69 |

## Ethical Considerations
The dataset was donated by Ronny Kohavi and Barry Becker to the UCI Machine Learning Repository. Site appropriately when using this dataset. Bcause this dataset includes demographic data such as gender, race, etc. some bias could be introduce if not used correctly. Be mindful of the effect this will have on your specific use case.

## Caveats and Recommendations
- The default hyperparameters were used. For improved metrics, hyperparameter tuning is recommended.  
- The data is extracted from 1994 census data and may no longer reflect earning potential in 2022.
