# LED-Digits-Recognition
Recognize LED digits by simple one-hidden layer of neural network

## Dataset
dataset.zip in folder is the original compressed source dataset. After uncompress, program would walk through each sub-folder (also the class) and randomly split them into train, dev and test dataset in temporary folder. Finally, all the saparated dataset would be converted into tfrecords to build a convenient data pipeline for tensorflow.

## Installation
1. Build a virtual environment to isolate from system-wide. Assuming the directory name is myproject  
`virtualenv -p python3 --no-site-packages myproject`
1. Change to the new directory  
`cd myproject`
1. Activate the established virtualenv  
`source bin/activate`
1. Get the source code from here and change to the directory  
`git clone https://github.com/jenhokuo/LED-Digits-Recognition.git`  
`cd LED-Digits-Recognition`
1. Install python3 dependent library  
`pip3 install -r requirements.txt`
1. Launch jupyter-lab and open **led_digits_recognition.ipynb**  
`jupyter-lab`

## Demo
Open **led_digits_recognition.ipynb**, just shift-enter to execute every cells
