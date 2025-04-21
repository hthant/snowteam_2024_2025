Jasem Al-Shukry         04-11-2025 (MM-DD-YYYY)
                        2025, April 11th

Machine Learning Code for Snowflake Classification

Purpose: 
    Identify snowflakes using machine learning, with many models and GPU testing being considered and experimented on. 
    The images of the snowflakes are first captured by the machine, then are passed on the S3 and matching code (separate 
    software) to crop and sharpen them. Once they are finished, they get pre-processed by the toKeras3 (this workspace) file to 
    ensure that TensorFlow can properly and effectively use them for machine learning training and testing.
    End goal is to achieve an acceptable program that outperforms original model by Hein Thant through various
    hyperparameter sweeping techniques.

How to run the code:
    IMPORTANT: Before executing any files, make sure to run "conda activate mascDB" first. This is because that the "mascDB"
    environment has all the necessary Python files to allow training of the machine learning algorithm. Without doing so, some
    libraries may not compile / execute as intended.

    Type the following lines into the terminal, depending on the options you want. You can also choose to create a .sh (bash)
    file that executes these commands sequentially (Look at GPU_test.sh for example):
    - "python mirrored.py":
        Runs "mirrored.py", outputs to terminal.
    - "python mirrored.py &> log-mirrored.txt":
        Runs "mirrored.py", prints output to "log-mirrored.txt".
    - "python mirrored.py &> log-mirrored.txt & disown":
        Runs "mirrored.py" in background, prints output to "log-mirrored.txt". Allows for other terminal operations
        without interrupting program execution.
    - "CUDA_VISIBLE_DEVICES=n python mirrored.py":
        Runs "mirrored.py" with GPU n being the only usable GPU to code, where n is 0, 1, or 2 depending on the chosen GPU.

    Useful commands for terminal usage:
    - "bash example.sh":
        Runs the bash script example.sh, much like the Python executions above. This is helpful when wanting to execute
        multiple terminal commands sequentially.
    - "watch -n n1 tail -n n2 log-mirrored.txt"
        Outputs real-time updates of log-mirrored.txt, where n1 is the refresh rate (seconds) and n2 is the number of
        lines visible per refresh. For example, n1 = 3 and n2 = 10 shows the last 10 lines of the file ever 3 seconds.
    - "nvtop"
        Tracks GPU and CPU usage. Helpful to check if GPUs are selected properly. Press the Q key to exit.
    - "btop"
        Tracks GPU and CPU usage, showing currently run programs. Helpful to check if program is actually running or has
        unexpectedly stopped. To forcibly stop a program, select it and press the K key to kill it. Press Q to exit.

Important Files:
    - utils3.py:
        Low-level utility code concerning database strcuturing.
    - toKeras3.py: 
        High-level utility code concerning dataset configuration.
    - mirrored.py: 
        Base code for training ML model with Mirrored Strategy configuration for GPUs. 
        Useful for hyperparameter sweeping or testing aspects of the code.
    - central.py:  
        Identical to mirrored.py, but uses Central Storage Strategy for GPU configuration.
    - log-mirrored.txt:
        Text file to show output of mirrored.py, used in hyperparameter sweeping for output analysis.
    - log-central.txt:
        Similar to log-mirrored.txt, but for central.py.
    - GPU_test.sh:
        Bash script for running 2 Python files (particularly mirrored.py and central.py) sequentially and sending their
        respective outputs to each of log-mirrored.txt and log-central.txt.

Issues and Solutions:
    Issue:    "train" method in "mirrored.py" does not work as intended. Error message is the following:
              "Input to reshape is a tensor with 64800000 values, but the requested shape has 8640000 [Op:Reshape]"
    Solution: "BATCHSIZE" variable in "mirrored.py" and "toKeras3.py" must match

    Issue:    OOM (Out Of Memory) can occur when setting "BATCHSIZE" to a number above the limit
    Solution: "BATCHSIZE" has to be <= 68 per GPU for InceptionV3 model, so maximum for 3 GPUs running simultaneously is 204.
              For other base models, refer to the following equation to calculate maximum theoritcal batch size:
              Max batch size = available GPU memory bytes / 4 / (size of tensors + trainable parameters)
              Reference: https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size
