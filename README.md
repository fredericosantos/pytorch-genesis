# 🚧 Under construction 🚧

<div align="center"> 

 
 <img alt="PyTorch Genesis" src="https://github.com/fredericosantos/pytorch-genesis/blob/master/pytorch_genesis_logo_text.png" width="800px" style="max-width: 100%;">

[![Paper](https://img.shields.io/badge/paper-doi.org%2F10.1016%2Fj.asoc.2023.110767-red)](https://authors.elsevier.com/sd/article/S1568494623007858)

</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{Frederico J.J.B. Santos, 
  title={Neuroevolution with box mutation: An adaptive and modular framework for evolving deep neural networks},
  author={Ivo Gonçalves, Mauro Castelli},
  journal={Applied Soft Computing},
  year={2023}
}
```   
