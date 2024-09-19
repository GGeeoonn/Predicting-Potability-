# I. Project Title
## Predicting Water Potability Using Machine Learning
### :Ensuring Safe Drinking   

<br><br/>
# II. Project Introduction
## Goal
Develop a machine learning model to predict water potability based on various water quality indicators.
## Motivation
**Public Health Improvement**<br/>
Access to clean drinking water is a fundamental right for all and a critical aspect of public health. However, many regions still struggle to access safe drinking water.
<br><br/>
**Environmental Protection**<br/>
Water pollution is closely linked to environmental degradation, and contaminated water can cause severe harm to both humans and natural ecosystems.
<br><br/>
**Sustainability**<br/>
The causes of pollution and water quality vary across regions, necessitating tailored management strategies that are adapted to the specific water characteristics of each area.
<br><br/>

# III. Dataset Description
## Water Potability Dataset
In this project, a total of 3,276 data samples are divided into Training set, Validation set, and Test set for use in model development.
<br><br/>
### Training Set:
Composed of 1,608 samples.
This dataset is used to train the machine learning model.
The model learns the relationship between various water quality indicators and potability during this stage.
### Validation Set:
Composed of 201 samples.
This set is used to evaluate model performance during training, allowing for tuning of hyperparameters and preventing overfitting.
It helps monitor the model’s generalization performance and adjust the learning process accordingly.
### Test Set:
Composed of 202 samples.
After the model is fully trained, this set is used to assess the final performance of the model.
It provides an evaluation of how well the model predicts water potability in real-world scenarios.
<br><br/>
This data splitting is essential for accurately assessing the model’s generalization performance and preventing overfitting or underfitting during the training process.
<br><br/>

## Dataset Composition
### Input Data
#### ph
pH measures the acidity or alkalinity of water.
<br/>(WHO recommended permissible limit: 6.5~8.5)
#### Hardness
Hardness is determined by the concentration of calcium and magnesium salts in the water, and water with high hardness may require additional treatment before consumption.
#### Solids(Total Dissolved Solids)
Solids represent the concentration of dissolved inorganic and organic substances in the water.
<br/>(WHO recommended permissible limit: ~1000 mg/L)
#### Chloramines
Chloramines are compounds formed when ammonia is added to chlorine for water disinfection.
<br/>(WHO recommended permissible limit: ~4 mg/L)
#### Sulfate
Sulfate occurs naturally and is found in various geological formations.
<br/>(WHO recommended permissible limit: 3~30 mg/L)
#### Conductivity
Conductivity measures the ability of water to conduct electricity, which is related to the concentration of ions dissolved in the water.
<br/>(WHO recommended permissible limit: ~400 μS/cm)
#### Organic_carbon
Organic Carbon indicates the total amount of carbon present in organic compounds in the water.
<br/>(WHO recommended permissible limit: ~2 mg/L)
#### Trihalomethanes
Trihalomethanes are compounds that can be found in water treated with chlorine.
<br/>(WHO recommended permissible limit: ~80 ppm)
#### Turbidity
Turbidity is determined by the amount of suspended solids in the water and indicates the cloudiness of the water.
<br/>(WHO recommended permissible limit: ~5 NTU)
<br><br/>
### Target Data
#### Potability
Potability is represented by a binary variable, where 1 indicates the water is safe for human consumption, and 0 indicates the water is unsafe to drink.

## ※ The docs folder also contains PowerPoint presentations.

