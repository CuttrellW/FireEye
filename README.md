<div>

  <br>
  <h2 align="center">FireEye</h2>

  <p align="center">
    Using Python and the Tensorflow Library to Detect Fires Visually with Machine Learning
    <br><br><br>
  </p>
</div>




<!-- ABOUT THE PROJECT -->
## About The Project


FireEye is a machine learning application 
designed to detect the presence of fire. It captures 
frames from video input and analyzes them against a training 
model to make accurate predictions to determine if the frame 
contains a fire.

Smoke detector technology hasn't changed much since they became commercially available in 1965, and with alarming increase in forest fires, the need for versatility and new methods of detecting fires in remote locations has grown more pertinent than ever. 
There are several advantages that ai-backed image processing can have over traditional smoke detectors.

Here are a few:
* Traditional smoke detectors are limited to a short proximity and closed environments
* Visual detection means that a properly trained AI can detect the presence of fires anywhere a camera can see
* Detection programs can be integrated into existing visual infrastructure (i.e. surveillance systems, satellite imagery)

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Tensorflow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To set up this project locally, you will need to download the fire dataset from Kaggle.com
[Downlaod the Dataset Here](https://www.kaggle.com/phylake1337/fire-dataset)

### Prerequisites

You will need Python 3 to run this program, as well as the package manager `pip3`. Note that Python 3.4 and later includes `pip3` by default.
To check your python version use the following command:
  ```sh
  python --version
  ```

### Installation


1. Clone the repo
   ```sh
   git clone https://github.com/CuttrellW/FireEye.git
   ```
2. Install requirements from included requirements.txt (Use `pip3` if using Python 3)
   ```sh
   pip3 install -r requirements.txt
   ```
3. Copy or move the downloaded fire dataset into the root directory of the project. Example folder structure:
   ```
    └── FireEye/
        ├── fire_dataset/
        │   ├── fire_images/
        │   └── non_fire_images/
        ├── src/
        │    │
        ...  ...

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


All configuration for running the program from start to finish can be found in the src/config/ folder. You can also create and save new config files and load it by changing the `DATASET_PATH` variable at the beginning of fire_eye.py.
You can use the configuration variables to make adjustments to the dataset and training model in order to optimize accuracy as well as change what happens when a fire is detected.

The program will train a new model on each run, then you can either show the statistics or begin the watch on your default camera device.

Selecting the plot option will historical values of training and validation accuracy and loss over every epoch.

When a fire is detected, the view window will close and the configured action will be taken (just a print message by default).
To end a watch early, press escape (make sure view window is selected).


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing


If you have a suggestion that would make this better, please fork the repo and create a pull request. Thank you for checking out my project!


<p align="right">(<a href="#top">back to top</a>)</p>




<!-- CONTACT -->
## Contact

Feel free to reach out for anything regarding this project or others!

Billy Cuttrell - wcuttrell@gmail.com

Project Link: [https://github.com/CuttrellW/FireEye](https://github.com/CuttrellW/FireEye)

<p align="right">(<a href="#top">back to top</a>)</p>


