##  Cross Comparison Representation Learning for Semi-supervised Segmentation of Cellular Nuclei in Immunofluorescence Staining




## Installation
* Install Pytorch 1.11.0 and CUDA 10.0
* Clone this repo
```
git clone https://github.com/rxy1234/CCRL
cd CLSD
```

## Data Preparation
* Download [CRAG dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet) <br/>
* Put the data under `./data/`
* Divide the dataset into six folders `myTest_Data ` `myTest_Label ` `myTraining_Data ` `myTraining_Label ` `myValid_Data ` `myValid_Label `
* Fill the folder `myTest_Label `with images with pixel values of all 0 corresponding to the folder `myTest_Data `

## Train
* Open `ema_train.txt ` in the folder `scripts `
* Copy content after modifying parameters
* Open the terminal and  `cd   'the path of the current project' `
* Paste the copied operation command and run it

## Evaluate
* Download [Test dataset](https://pan.baidu.com/s/16VGSDwuCUk1QDT4AUcyMLg?pwd=w2zv ) <br/>
* Put the test data under `./data/`
* Download [Parameter](https://pan.baidu.com/s/1I643oztfFyZJXXKUW4IIRg?pwd=qbxt ) <br/>
* Put the parameter under `./exp/`
* Open `ema_test.txt ` in the folder `scripts `
* Copy content after modifying parameters
* Open the terminal and  `cd   'the path of the current project' `
* Paste the copied operation command and run it

## Acknowledgement
Some code is reused from the [Pytorch implementation of mean teacher](https://github.com/CuriousAI/mean-teacher). 

