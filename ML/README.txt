Machine Learning Code for Snowflake Classification

Purpose: 
    Identify snowflakes using machine learning, with many models and GPU testing being considered and experimented on.
    End goal is to achieve an acceptable program that outperforms original model by Hein Thant through various
    hyperparameter sweeping techniques.

Current Issue:
    "train" method in "mirrored.py" does not work as intended. Error message is the following:
    "Input to reshape is a tensor with 64800000 values, but the requested shape has 8640000 [Op:Reshape]"
    Will update if problem is resolved with solution in case it happens in the future
