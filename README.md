# Papaya Disease Classification using CNN

This project aims to classify papaya diseases using a Convolutional Neural Network (CNN). It trains the model to recognize three types of diseases: Anthracnose, Phytophthora, and Ring Spot using image data.

## Project Structure

* papaya_cnn.ipynb: Jupyter notebook containing the code for loading images, preprocessing them, building the CNN model, and training the model on the dataset.
* Input_data: Directory containing the papaya disease images categorized into subfolders.
* model.h5: Pre-trained model for predicting papaya diseases (if available).

## Installation
1. Clone the repository.

2. Install required dependencies:

   <code> pip install tensorflow keras opencv-python matplotlib </code>
3. Ensure that the dataset is stored in the Input_data directory. Each disease type should have its folder, and images should be inside these folders.

## Dataset
The dataset consists of images of papaya leaves affected by the following diseases:

* Anthracnose
* Phytophthora
* Ring Spot
  
Each folder in the dataset contains images of papaya leaves with one of these diseases.

## Usage

1. Training the Model: To train the model, load the dataset, and run the cells in the papaya_cnn.ipynb notebook.

* The model is a simple CNN with multiple convolutional layers followed by dense layers for classification.
* You can adjust the hyperparameters such as the learning rate, number of epochs, or batch size in the notebook.
  
2. Testing on a New Image: You can test the trained model on a new image using the following approach:

   `
image_path = 'Input_data/test_image.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (80, 80))  # Resize to match the input size
image = image / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Add batch dimension

predictions = model.predict(image)
print(predictions)  # The output will show the predicted class
`

## Model
The Convolutional Neural Network (CNN) model used here has the following layers:

1. Convolutional layers with ReLU activation
2. MaxPooling layers
3. Flatten and Dense layers for final classification
4. Output layer using a softmax or sigmoid activation depending on the number of classes.
   
## Results

After training, the model can classify the papaya disease from the input images. You can monitor the performance using accuracy and loss plots during training.

## Contributions

Feel free to fork this repository and submit pull requests for improvements!

Thank you, Ashen
