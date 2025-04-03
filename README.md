# 2024-Election-Prediction

After the results of the 2024 US federal election, I wanted to take a closer look at the socio-economic factors that played a role into voting behavior.  I thought it would be fun to make a predictive model of voting behavior based on some of the available county-level data sets to see which party each county would vote for. Infomration on how the data was gathered and prepared is in it's own file named "data cleaning & acquisition". 

I used three machine learning models to make predictions; Logistic Regression, k-nearest neighbour and Random Forest. The strongest one after optimizing hyper-parameters was a Random Forest model with a final accuracy of around 93% and an ROC AUC score of ~95%. 
