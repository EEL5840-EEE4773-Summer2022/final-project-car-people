

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/catiaspsilva/README-template">
    <img src="images/group-logo.jpg" alt="Logo" width="150" height="150">
  </a>
  </p>
</p>

# Car People Final Project
# Authors Michelle Dupuis & Everett Periman

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this section you should describe your project, including any datasets you used and appropriate citations. You may refer to your project report or cite your paper for more detailed information.

[Here goes the title with hyperlink](https://github.com/catiaspsilva/README-template)

You can include tables or images to summarize your results when and if appropriate.

<!-- GETTING STARTED -->
## Getting Started

In this section you should provide instructions on how to use this repository to recreate your project locally.

### Dependencies

Python >= 3.10.2

tensorflow == 2.9.1

scikit-learn == 1.1.1

numpy == 1.22.3




### Automated dependency installs 

requirements.txt is included for easier importing of the required python packages into your workspace
enviroment.yaml is included for easier importing of the conda env into your workspace

```sh
pip install -r requirements.txt
  ```
  
```sh
conda env create -f enviroment.yml
  ```
### Installation

1. Clone the repo
```sh
git clone https://github.com/EEL5840-EEE4773-Summer2022/final-project-car-people.git
   ```
   
3. Setup (and activate) your environment
```sh
conda env create -f enviroment.yml
  ```
  
  ```sh
conda activate fml
  ```
  
4. Install required dependencies either manually or by running
```sh
pip install -r requirements.txt
  ```
  
<!-- USAGE EXAMPLES -->
## Usage

### FinalProject.ipynb: This is used to show the development process and the general tools that were used in this project. Many small sections are included to help the team visualize what the layers represent or show the effects of data augmentation. This is included for completion and context for the next 3 files.

### train.py: This is a procedural Python script that will train the model to fit the training data.
To run the script use this command line
```sh
python train.py
  ```
  The data and label input files can be changed with the following variables at the top of the script
 ```sh
IMPORT_DATA_FILENAME = 'training_data.npy'
IMPORT_LABELS_FILENAME = 'training_labels.npy'
  ```
  The model output filename can be changed with the following variable at the top of the script
NOTE: Ensure that this filename is transferred to the testing scripts for proper operation.
```sh
H5_OUPUT_FILENAME = "inception_v3_trained.h5"
  ```
  
 ### test.py: This is a procedural Python script that will test the performance of the model on the EASY dataset. This script will print the accuracy and predicted labels to the terminal and the predicted labels will also be stored in the PREDICTED_LABEL_OUTPUT_FILENAME file.
```sh
python test.py
  ```
 The data and label input files can be changed with the following variables at the top of the script
 ```sh
IMPORT_DATA_FILENAME = 'easy_testing_data.npy'
IMPORT_LABELS_FILENAME = 'easy_testing_labels.npy'"
  ```
If the H5 output filename was changed in the previous step and a new model was trained change the name in this variable at the head of the script.
```sh
H5_INPUT_FILENAME= "inception_v3_trained.h5"
  ```
 The predicted labels filename can be changed with the following variables at the top of the script
```sh
PREDICTED_LABEL_OUTPUT_FILENAME = 'predicted_labels.npy'
  ```
 The THRESHOLD value should NOT be altered.
```sh
THRESHOLD = 0
  ```
  
   ### hard_test.py: This is a procedural Python script that will test the performance of the model on the HARD dataset. This script will print the accuracy and predicted labels to the terminal and the predicted labels will also be stored in the PREDICTED_LABEL_OUTPUT_FILENAME file.
To run the script use this command line
```sh
python hard_test.py
  ```
 The data and label input files can be changed with the following variables at the top of the script
 ```sh
IMPORT_DATA_FILENAME = 'hard_testing_data.npy'
IMPORT_LABELS_FILENAME = 'hard_testing_labels.npy'"
  ```
If the H5 output filename was changed in the previous step and a new model was trained change the name in this variable at the head of the script.
```sh
H5_INPUT_FILENAME= "inception_v3_trained.h5"
  ```
 The predicted labels filename can be changed with the following variables at the top of the script
```sh
PREDICTED_LABEL_OUTPUT_FILENAME = 'hard_predicted_labels.npy'
  ```
 The THRESHOLD value should NOT be altered.
```sh
THRESHOLD = 0.8071025020177562 
  ```
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- Authors -->
## Authors

Michelle's Profile - [Michelle Dupuis Webcourses](https://ufl.instructure.com/courses/455013/users/1128233)
Everetts Profile - [Everett Periman Webcourses](https://ufl.instructure.com/courses/455013/users/1164029)

Project Link: [EEL5840-EEE4773-Summer2022/final-project-car-people](https://github.com/EEL5840-EEE4773-Summer2022/final-project-car-people)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

You can acknowledge any individual, group, institution or service.
* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)

## Thank you

<!-- If this is useful: [![Buy me a coffee](https://www.buymeacoffee.com/assets/img/guidelines/download-assets-sm-1.svg)](https://www.buymeacoffee.com/catiaspsilva) -->
