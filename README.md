# Sound-Classifier

In this project, I am classifying 10 different sounds. This project was inspired by Krish Naik's YouTube video. The dataset is publically available from <a href = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"> here </a>
<br>
<b>Steps</b>
1. Download  the dataset <br>
 `!wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz`
2. Go to the directory of downloaded dataset and extract using <br>
  `!tar -xf <file_name>` <br>
3. To train the model, use command as, <br>
  `!python train.py --checkpoint <location where you want to save the weights> --epochs <epoch number> --dataset_dir <directory of dataset>`
<br>
<b>Implementation of predict.py is not complete yet.</b><br>

The accuracy obtained by Sound_Classifier.h5 is 92% on the test data. <br>
