# Human_detection_with_live_web_streaming_using_Opencv_Tensorflow_Flask
This is a python code that uses machine learning to detect humans in real time and stream the output as video in a simple Flask web application. It utilises OpenCV, Tensorflow, and Flask libraries.The machine learning model used here is pre-trained SSD MobileNet V3 model.
## Prerequisites
Before running the code in your system, make sure you have the following installed:
* Python 3.x
* TensorFlow
* OpenCV
* Flask
### Note
The code doesn't explicitly use TensorFlow. However, it indirectly relies on TensorFlow as the SSD MobileNet V3 model is typically trained and exported using TensorFlow. So running the code without properly installing TensorFlow might give you error messages.
## Getting started
1. Download the frozen inference graph.pb [here](frozen_inference_graph.pb) and put in the same folder where the main.py is saved.
2. Download the all the files or clone this repository.
3. Run main.py
4. Open your web browser and go to http://127.0.0.1:5000/ to view the human detection in local web servor.
