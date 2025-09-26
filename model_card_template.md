# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained to predict whether a person’s income exceeds $50K per year based on census data. The model uses categorical features like workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Intended Use
The model is intended for educational purposes and to demonstrate a machine learning pipeline. It predicts income level from demographic data.

## Training Data
The model was trained on 80% of the census dataset (`census.csv`) which includes features such as workclass, education, marital-status, occupation, relationship, race, sex, and native-country.

## Evaluation Data
The remaining 20% of the census dataset was used as test data to evaluate the model's performance.

## Metrics
The model was evaluated using **Precision, Recall, and F1-score**. Overall test performance:
- Precision: 0.7391
- Recall: 0.6384
- F1: 0.6851

Sample performance on data slices by feature:

**Workclass slices:**
- Federal-gov, Count: 191 — Precision: 0.7971 | Recall: 0.7857 | F1: 0.7914
- Local-gov, Count: 387 — Precision: 0.7500 | Recall: 0.6818 | F1: 0.7143
- Private, Count: 4,578 — Precision: 0.7362 | Recall: 0.6384 | F1: 0.6838

**Education slices:**
- Bachelors, Count: 1,053 — Precision: 0.7569 | Recall: 0.7333 | F1: 0.7449
- Masters, Count: 369 — Precision: 0.8263 | Recall: 0.8502 | F1: 0.8381
- HS-grad, Count: 2,085 — Precision: 0.6460 | Recall: 0.4232 | F1: 0.5114

## Ethical Considerations
The model may inherit biases present in the training data. Factors like race, sex, and native-country are included, which could unintentionally impact predictions.

## Caveats and Recommendations
This model is meant for demonstration and learning. It should not be used in real-world decision-making without careful evaluation and bias mitigation.
