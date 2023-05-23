# BPQP
The implementation of the paper: "An Efficient Differentiable Convex Optimization  Framework for End-to-End Learning in Decision Making" [TODO: arXiv Hyperlink]

![avatar](frame.png) 

# Data & Environment
* Install python3.7, 3.8 or 3.9.
* Install the requirements in [requirements.txt](requirements.txt).
* Install the quantitative investment platform Qlib and download the data from Qlib:
```
# install Qlib from source
pip install --upgrade  cython
git clone https://github.com/microsoft/qlib.git && cd qlib
python setup.py install

# Download the stock features of Alpha158 from Qlib
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2 
```
* Run [dataset/papare_dataset.py](dataset/prepare_dataset.py) to generate train/valid/test dataset
```
python papare_dataset.py
```
# Reproduce our BPQP in large-scale QPs and LPs experiment

![avatar](speed.png) 
[Large scale QPs and LPs experiment.ipynb](Large scale QPs and LPs experiment.ipynb)


# Reproduce our COFFE framework for end-to-end portfolio optimization
```
python main.py --market CN --loss huber_loss --predictor mlp --solver bpqp
```

# Reproduce benchmark
* Two-Stage
```
python main.py --market CN --loss mse --predictor mlp --solver bpqp
```
* Naive NN
```
python main.py --market=CN --loss=huber_loss --predictor=mlp --solver=naive_nn  
```
* DC3
```
python main.py --market=CN --loss=huber_loss --predictor=mlp --solver=dc3  
```