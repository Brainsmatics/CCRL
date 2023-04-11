##  CLSDNet：Simple Contrastive Representation Learning for Semi-supervised  Segmentation of Cellular Nuclei in Immunofluorescence Staining  




## Installation
* Install Pytorch 1.11.0 and CUDA 10.0
* Clone this repo
```
git clone https://github.com/xmengli999/TCSM
cd TCSM
```

## Data Preparation
* Download [CRAG dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet) <br/>
* Put the data under `./data/`
* Divide the dataset into six folders
* Fill the folder `myTest_Label `with images with pixel values of all 0 corresponding to the folder `myTest_Data `

## Train
* Open `ema_train.txt ` in the folder `scripts `
* Copy content after modifying parameters
* Open the terminal and  `cd   'the path of the current project' `
* Paste the copied operation command and run it

## Evaluate
* Open `ema_test.txt ` in the folder `scripts `
* Copy content after modifying parameters
* Open the terminal and  `cd   'the path of the current project' `
* Paste the copied operation command and run it

## Acknowledgement
Some code is reused from the [Pytorch implementation of mean teacher](https://github.com/CuriousAI/mean-teacher). 

