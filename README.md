# nlp-final-project
Model trained in ``Python 3.10.12`` -
Ensure that all these libraries are installed. If not please do so and then run the file -
``math
torch
d2l
pandas
csv
os
matplotlib``

# For Google Colaboratory
1. In Google Colab environment, ``run %pip install d2l``.
2. Set runtime to t4 GPU and restart the runtime.
3. Put _new_approach.py_ and _test_data_movie.csv_ and _encoder_block_model.pt_ (the pretrained model weights) in the "Files" section on Google Colaboratory.
4. The pretrained model weights should be present in the zip file of the repository.
   
   If the weights are present,
      - Run “_python new_approach.py_”

   The user will be prompted to input “train” or “infer”. Enter “infer” and continue.

   Observe model results.
   
   If not, or if you wish to train again,
   - Run “_python new_approach.py_”

   The user will be prompted to input “train” or “infer”. Enter “train” and continue.

   The model will train and run inference on the IMDB test set as well as the provided test set and return the evaluation metrics.

   The new weights will be saved and can be used for inference.

# For Local
1. Download the zip file
2. Install all relevant modules using `pip install math torch d2l pandas csv os matplotlib`, in Python 3.10.12
4. The pretrained model weights should be present in the zip file.
5. Follow the below steps with a terminal open in the folder containing the files.
6. If the weights are present,
      Run “_python new_approach.py_”
      The user will be prompted to input “train” or “infer”. Enter “infer” and continue
      Observe model results
7. If the weights are not present, or if you wish to train again,
      Run “_python new_approach.py_”
      The user will be prompted to input “train” or “infer”. Enter “train” and continue
      The model will train and run inference on the IMDB test set as well as the provided test set and return the evaluation metrics.
      The new weights will be saved and can be used for further inference.
