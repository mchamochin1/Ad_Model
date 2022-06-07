# Prediction of Sales from Marketing Actions
A national distribution company intends to use a model developed by the data science department, with which they obtain a prediction of sales from the marketing expenses of advertisements on television, radio and newspapers. They want to incorporate this data into their internal web page, where they share all kinds of information related to company results, sales, acquisitions, etc... The web is developed in AngularJS, while the model was developed in Python, so that we need a communication interface between both systems.

The development team needs you to implement a microservice so that they can consume the model from the web itself. The microservice must meet the following characteristics:
1. Offer sales prediction based on all ad spend values. (/api/v1/predict)
2. Possibility of retraining the model again with the possible new records that are collected. (/api/v1/retrain)

It deploys a machine learning model to an API for consumption in Pythonanywhere. It can train the model, save it trained, and it ables to use an API that allows consuming said model from any other technology.

The model is:
1. Deployed from a github repository.
2. Checked concerning paths to load it.
3. Provided with a Python script (*train.py*) which is the training code, with the design of an API with Flask .
