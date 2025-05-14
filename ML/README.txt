Jasem Al-Shukry         05-05-2025 (MM-DD-YYYY)
jasemmfs@gmail.com      2025, May 5th

Machine Learning Code for Snowflake Classification

Purpose: 
    Identify snowflakes using machine learning, with many models and GPU testing being considered and experimented on. 
    The images of the snowflakes are first captured by the machine, then are passed on the S3 and matching code (separate 
    software) to crop and sharpen them. Once they are finished, they get pre-processed by the toKeras3 (this workspace) file to 
    ensure that TensorFlow can properly and effectively use them for machine learning training and testing.
    End goal is to achieve an acceptable program that outperforms original model by Hein Thant through various
    hyperparameter sweeping techniques.

Connecting to the GPU server:
    Make sure you're in the CSU VPN (for remote working, download GlobalProtect via CSU's websites). In Visual Studio, type 
    "ssh snow@129.82.224.175" with password "Snowcam" when prompted. You'll mostly be working in the "snow/smij" subdirectory,
    but the rest of the files are used implicitly / are for reference.

How to run the code:
    IMPORTANT: Before executing any files, make sure to run "conda activate mascDB" first. This is because that the "mascDB"
    environment has all the necessary Python files to allow training of the machine learning algorithm. Without doing so, some
    libraries may not compile / execute as intended.

    Type the following lines into the terminal, depending on the options you want. You can also choose to create a .sh (bash)
    file that executes these commands sequentially (Look at GPU_test.sh for example):
    - "python mirrored.py":
        Runs "mirrored.py", outputs to terminal.
    - "python mirrored.py > log-mirrored.txt":
        Runs "mirrored.py" and prints output to "log-mirrored.txt".
    - "python mirrored.py &> log-mirrored.txt":
        Runs "mirrored.py" in background (indicated by the '&'), prints output to "log-mirrored.txt".
    - "python mirrored.py &> log-mirrored.txt & disown":
        Runs "mirrored.py" in background, prints output to "log-mirrored.txt". 'disown' allows command to run even if
        user is logged out (i.e. helpful if you need to do something else but can't keep an eye on execution).
    - "CUDA_VISIBLE_DEVICES=n python mirrored.py":
        Runs "mirrored.py" with GPU n being the only usable GPU to code, where n is 0, 1, or 2 depending on the chosen GPU.
    - "./mirrored.py"
        If "mirrored.py" has "#![insert Python compiler path here]" then this command runs the script using the Python
        compiler, as opposed to the above example commands. I personally didn't use this, however if you feel this easier
        for you, you're more than welcome to edit the files to implement this convention.

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
    Any files in the "smij" directory that is not in a sub-directory and also not mentioned here are either
    intuitive, simple, or are available for reference (not actively being used).
    - utils3.py:
        Low-level utility code concerning database strcuturing.
    - toKeras3.py: 
        High-level utility code concerning dataset configuration, particularly ensuring that the dataset used for training
        the algorithm is compatible with TensorFlow.
    - mirrored.py: 
        Base code for training ML model with Mirrored Strategy configuration for GPUs. This is the main file that you will be 
        using to train the ML model.

Issues and Solutions:
    Issue:    "train" method in "mirrored.py" does not work as intended. Error message is the following:
              "Input to reshape is a tensor with 64800000 values, but the requested shape has 8640000 [Op:Reshape]"
    Solution: "BATCHSIZE" variable in "mirrored.py" and "toKeras3.py" must match

    Issue:    OOM (Out Of Memory) can occur when setting "BATCHSIZE" to a number above the limit
    Solution: "BATCHSIZE" has to be <= 68 per GPU for InceptionV3 model, so maximum for 3 GPUs running simultaneously is 204.
              For other base models, refer to the following equation to calculate maximum theoritcal batch size:
              Max batch size = available GPU memory bytes / 4 / (size of tensors + trainable parameters)
              Reference: https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size
