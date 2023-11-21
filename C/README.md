# OKM, OKMED and WOKM in C language

This is the implementation used for the comparison in the paper:  
Study of Overlapping Clustering Algorithms Based on Kmeans through FBcubed Metric  
https://doi.org/10.1007/978-3-319-07491-7_12

## Introduction

The `input` folder has the 3 datasets used in the comparison from the paper, notice that the header is different from the original repository  
The `source` folder contains the implementation of the 3 algorithms in C, also fbcubed and fmeasure metrics are included since theese metrics were used in the comparison  

## Input header

The header must contain the next format:  
  
2407 -> Number of instances or rows to be readed from the dataset  
294 -> Number of features or attributes, without incluiding classes or labels  
6 -> Number of classess or labels, since is a multilabel dataset the lasts c columns will be taken as theese classes or labels  
0,0,0,0,2,2,2,2,2,2 -> 0 for each numeric column, otherwise the number of distinct possible values from that column (not numeric). In the case of the classes columns the value must be 2 since it can be 0 or 1 (0 - does not belong to that class, 1 belongs to that class)  
145.52,5,4656,325,0,1,0,0,0,1 -> The data  
  
Not numeric values must be encoded as consecutive numeric values since there is no support for characters (support in next version is coming but it will be in C++ and python). Example, column with 3 discrete values 'small', 'medium', 'big' must be converted to 0, 1, 2 with each value corresponding to each string.

## Build

Compile the C code using the following commands:

```bash
gcc source/okm.c -lm -o okm.o
gcc source/okmed.c -lm -o okmed.o
gcc source/wokm.c -lm -o wokm.o
gcc source/fbcubed.c -lm -o fbcubed.o
gcc source/fmeasure.c -lm -o fmeasure.o
```

## Usage
```bash
okm.o data.csv k seed(optional)
```
being `data.csv` the file containing the data to process  
`k` the number of clusters to find  
`seed` the seed value used for the random generated values  

```bash
okmed.o data.csv k seed(optional)
```
being `data.csv` the file containing the data to process  
`k` the number of clusters to find  
`seed` the seed value used for the random generated values  

```bash
wokm.o data.csv k beta seed(optional)
```
being `data.csv` the file containing the data to process  
`k` the number of clusters to find  
`beta` the value for beta parameter (must be > 1)  
`seed` the seed value used for the random generated values  

## Disclaimer
This code is provided "as is" without warranty. The author is not responsible for any malfunction, damage, or loss of data caused by the use of this code.

## Acknowledgements
If you use or modify this code for your own projects or research, please acknowledge the author.