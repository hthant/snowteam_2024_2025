Jasem Al-Shukry         02-14-2025 (MM-DD-YYYY)
                        2025, February 14th

Machine Learning Code for Snowflake Classification

Purpose: 
    Identify snowflakes using machine learning, with many models and GPU testing being considered and experimented on.
    End goal is to achieve an acceptable program that outperforms original model by Hein Thant through various
    hyperparameter sweeping techniques.

Issues and Solutions:
    Issue:    "train" method in "mirrored.py" does not work as intended. Error message is the following:
              "Input to reshape is a tensor with 64800000 values, but the requested shape has 8640000 [Op:Reshape]"
    Solution: "BATCHSIZE" variable in "mirrored.py" and "toKeras3.py" must match

    Issue:    OOM (Out Of Memory) can occur when setting "BATCHSIZE" to a number above the limit
    WIP:      Seems like BATCHSIZE=64 per GPU used is a good number, but OOM occured with 192 for 3 GPUs
