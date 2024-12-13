# nlp-final-project
Ensure that all these libraries are installed. If not please do so and then run the file
math
torch
d2l
pandas
csv
os
matplotlib


1. In Google Colab environment, run %pip install d2l
2. Set runtime to t4 GPU 
3. Restart runtime
4. Put new_approach.py and test_data_movie.csv in the files section on colab.
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

