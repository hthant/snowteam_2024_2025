Jasem Al-Shukry         02-21-2025 (MM-DD-YYYY)
                        2025, February 21st

Machine Learning Code for Snowflake Classification

Purpose: 
    Identify snowflakes using machine learning, with many models and GPU testing being considered and experimented on.
    End goal is to achieve an acceptable program that outperforms original model by Hein Thant through various
    hyperparameter sweeping techniques.

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

Python files documentation:
    - utils3.py:
        Each function has its descrption commenetd above it.
    - toKeras3.py:
        § transformTensor(arr):
            Takes arr, casts its elements to 32-bit (single precision) floating point values, subtracts each value
            by the mean of all the elements, divides the results by the standard deviation of the elements, then reshapes
            the resulting object into a 4D array of size [3, 300, 300, BATCHSIZE].
            Separately, a variable 'out' is created, which is a 4D array of size [0, 300, 300, 3] filled with 0's.
            In the for loop that runs from 0 until BATCHSIZE - 1 inclusive, a 'temp' variable is created by taking the
            ith index in the 4th dimension of arr. Then, 3 copies of the 'temp' are stacked vertically, and reshaped into
            a [3, 300, 300, 3] 4D object, then concatenates it into the 'out' variable declared outsdie the loop.
            'out' is then reshaped into a 5D object of size [BATCHSIZE, 3, 300, 300, 3]. Finally, 'out1', 'out2', and 'out3'
            are created, which take the 1st, 2nd, and 3rd index respectively of the reshaped 'out', and reshape them again into
            a 4D object of size [BATCHSIZE, 300, 300, 3], which are then returned by the function.
        § transformLabel(l):
            'l' in this case is typically a list of labels. These can be found in utils3.py as:
            CLASSNAMES = ['small_particle', 'columnar_crystal', 'planar_crystal', 'aggregrate', 'graupel', 'columnar_planar_combination']
            RIMINGNAMES = ['unrimed', 'rimed', 'densely_rimed', 'graupel-like', 'graupel']
            Takes the values of 'l', subtracts them by 1, casts them into 8-bit unsigned integers, then one-hot codes them.
            Used for the labels in utils3.py.
            Refer to Tensorflow documentation below for tf.one_hot.
            https://www.tensorflow.org/api_docs/python/tf/one_hot?hl=en
        § class CustomTFDataset(tf.keras.utils.Sequence):
            wszj
            - __len__(self) -> int:
                Returns the length of the X_generator passed on to CustomTFDataset.
            - __getitem__(self, idx: int) -> Tuple[Any, Any]:
                Takes the X_generator and y_generator objects, converts them to tensors and assigns them to
                X_batch and y_batch respectively. X_batch is then passed on to transformTensor, and y_batch is 
                passed on to transformLabel. The function returns the tuple of the two results from these
                operations.
                Refer to Tensorflow documentation below for tf.convert_to_tensor:
                https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor

Issues and Solutions:
    Issue:    "train" method in "mirrored.py" does not work as intended. Error message is the following:
              "Input to reshape is a tensor with 64800000 values, but the requested shape has 8640000 [Op:Reshape]"
    Solution: "BATCHSIZE" variable in "mirrored.py" and "toKeras3.py" must match

    Issue:    OOM (Out Of Memory) can occur when setting "BATCHSIZE" to a number above the limit
    Solution: "BATCHSIZE" has to be <= 68 per GPU, so maximum for 3 GPUs running simultaneously is 204
