
# Image Binarization Application

This repository contains an application for image binarization, specifically designed for historical document images. The application utilizes advanced techniques based on Convolutional Neural Networks (CNNs) to achieve accurate and efficient binarization results. The backend of the application is implemented using Flask, while the frontend is built with Bootstrap.




## Screencast

![Screencast](https://drive.google.com/uc?export=view&id=1nnbd7fsuwqublFyEfEHJfuP1cQDtIjNF)

## Table of Contents

- Introduction
- Installation
- Optimizations
- Used By
- Authors
- References



## Introduction
The application offers a user-friendly interface for uploading document images and applying binarization techniques. Binarization is the process of converting a grayscale or color image into a binary image, where each pixel is represented as either black or white. This is particularly useful for enhancing the legibility and extracting text from historical documents.
## Installation

Clone the repository onto your local machine:
```bash
  git clone https://github.com/agniramadani/ImageBinarization.git
```
Create a virtual environment:
```bash
  python3 -m venv name
```
Activate the virtual environment for MacOS or Linux::
```bash
  . bin/activate
```
Activate the virtual environment for Windows:
```bash
  source name/bin/activate
```
[Download](https://drive.google.com/file/d/1A3QeiPwjQM2wUwMwyyWSgT9mzsEx4Q-T/view) DPLinkNet weights and save them inside the cloned app directory, you can follow these steps:
```bash
  cd /path/to/cloned/app
```
Unzip the file using the appropriate command:
```bash
  unzip dplinknet_weights.zip
```
**Important Note**: Please be aware that in the code, instead of using the name "dplinknet_weights", only "weights" is utilized. It is highly recommended to either rename the file "dplinknet_weights" to "weights" or make the necessary changes within the code.

To rename the file for MacOS or Linux: 
```bash
  mv dplinknet_weights weights
```
For Windows:
```bash
  ren dplinknet_weights weights
```
To install the required dependencies, run the following command:
```bash
  make install
```
Once the installation is complete, you can start the app by running the following command:
```bash
  make start
```



## Optimizations


The image binarization algorithm employed in this application has been specifically modified to cater to the unique requirements of historical document processing, particularly within the domain of ancient history. Unlike the original DP-LinkNet algorithm, which does not have inherent support for both CUDA and CPU platforms, the modified algorithm in this application has been optimized to ensure efficient performance on both hardware configurations. This adaptation offers users the flexibility to utilize the application on their preferred hardware, regardless of whether it is CUDA-enabled or CPU-based.



## Used By

The image binarization application is used by:

- [Dr. Isabelle Marthot-Santaniello](https://daw.philhist.unibas.ch/de/personen/isabelle-marthot-santaniello/)
- https://d-scribes.philhist.unibas.ch/


## Author
- [Agni Ramadani](https://github.com/agniramadani)


## References
[Here](https://github.com/beargolden/DP-LinkNet) you can explore further details about the DP-LinkNet algorithm, which serves as a key reference for the development of the image binarization capabilities in this application.