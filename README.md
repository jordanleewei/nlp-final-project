# nlp-final-project
Model trained in Python 3.10.12 -
Ensure that all these libraries are installed. If not please do so and then run the file -
math
torch
d2l
pandas
csv
os
matplotlib

# For Colab
1. In Google Colab environment, run %pip install d2l
2. Set runtime to t4 GPU 
3. Restart runtime
4. Put new_approach.py and test_data_movie.csv and encoder_block_model.pt (the pretrained model weights) in the files section on colab.
5. The pretrained model weights should be there in the zip file of the repository.
   
   If the weights are present,
      Run “python new_approach.py”
      An input will be prompted to train or infer. Enter “infer” and continue
      Observe model results
   
   If not, or if you wish to train again,
      Run “python new_approach.py”
      An input will be prompted to train or infer. Enter “train” and continue
      The model will train and run inference on the IMDB test set as well as the provided test set and return the evaluation metrics.
      The new weights will be saved and can be used for inference.

# For Local
Alternatively, the execution instructions are as below:
1. Download the zip file
2. Install all relevant modules using `pip install math torch d2l pandas csv os matplotlib`, in Python 3.10.12
4. The pretrained model weights should be there in the zip file.
5. Do the below steps with a terminal open in the folder containing the files.
6. If the weights are present,
      Run “python new_approach.py”
      An input will be prompted to train or infer. Enter “infer” and continue
      Observe model results
7. If not, or if you wish to train again,
      Run “python new_approach.py”
      An input will be prompted to train or infer. Enter “train” and continue
      The model will train and run inference on the IMDB test set as well as the provided test set and return the evaluation metrics.
      The new weights will be saved and can be used for inference.
