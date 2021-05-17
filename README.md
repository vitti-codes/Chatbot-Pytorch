# Chatbot-Pytorch-INM706
This repository contains the implementation of the final solution for the coursework of Deep Learning for Image Analysis [INM706] at City, University of London.

We implemented a Seq2Seq model to build a Chatbot. The model can be downloaded from this repository and in order to run the Jupyter Notebook and the python files you must use the virtual environment provided in the following folder.

The folder contains also the pretrained model as well as the checkpoints to continue the training phase from epoch 70.

Folder link:
https://cityuni-my.sharepoint.com/:f:/g/personal/tommaso_capecchi_city_ac_uk/EjdjQxfa9aFBk2pYi_6lTkcB10CEgxshXgrKTjNutuof-w?e=jEnQvX

In order to properly load the pretrained model, you have to create a folder inside the project root directory named 'saved_models' and drag-and-drop the .pth files inside this directory.

We implemented a simple GUI to easily interact with the Chatbot. By default the chatbot is loaded with the model trained with attention. If you wish to interact with the model without no attention you have to initialize the model by setting the property 'with_attention=False' during instantiation. Also you have to initialize the searcher for the evaluation phase with the property 'attention=False'.

# Launch GUI
To be able to execute the program and use the GUI you have to run the utils.py file. The GUI should appear immediately. If you stored properly the pretrained files in the corresponding folder, then the model with attention will be loaded and the answer the chatbot will give are output by this model.
