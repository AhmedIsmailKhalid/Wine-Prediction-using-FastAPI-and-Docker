## Deploying a Wine Prediction Model with FastAPI and Docker

This project deploys a webserver hosts a machine learning predictive model trained on the wine dataset using FastAPI and Docker. It builds and integrates code with Docker to making it portable and easy to deploy. The goal of this project is to deployment of a trained machine learning model using FastAPI and Docker container, hence as such project uses a simpler Random Forest classifier using standard scaler for scaling the features. The Random Forest classifier has 10 trees. Scikit-learn model is used rather than a TensorFlow or PyTorch model to avoid long build times as deep learning framework models have much longer build times when a Docker image is being build due to their size. 

**Note :** This project uses Docker 3.6.0 on Windows 10 Home  Edition 21H2 version (build 19044.2006) because the latest version of Docker would not start or work. If you are a windows user experiencing issues with latest Docker (especially with Docker Dameon not working or related pipeline errors, just download and install Docker 3.6.0). Also make sure that you are using WSL2 backend if you are using Windows (WSL 2 backend is installed with Docker 3.6.0 and the latest version of Docker)

## Why Deep Learning Frameworks Are Not Used?
TensorFlow and Pytorch are great and powerful frameworks for developing deep learning models. However, it is possible that the model being used for deployment isn't a Deep Learning model at all! Additionally, as mentioned above, the build times for deep learning mdoels can be longer due to their size. Furthermore, frameworks such as TensorFlow and Pytorch provide tools such as TFXServing and TorchServe to help deploy and serve the models. As such, this project explores using machine learning models lack such tools for deploying and serving models. Thus this project serves as a guide for newcomers giving them an insight to how web servers intergrate with models.

## Docker File
The FROM instruction allows selecting a pre-existing image as the base for the new image, thus all of the software available in the base image will also be available on the new image. This image contains an alpine version of Linux, a distribution very small in size. It also includes miniconda with Python 3.7.

## Building The Image(s)
This project has two images : one for serving predictions single data point at a time and one for serving batch predictions. In order to build the images make sure that the directory is set to the respective folders, i.e. `no-batch` and `with-batch`. Then use the following template to run build the image
```
docker build image-name:image-tag .
```
Replace the `image-name` with the desired name of the image and `image-tag` with the desired tag for the image. 


## Run the container(s)
Once the image(s) has been built next step is to run the image. This can be done using the following command :
```
docker run --rm -p 80:80 image-name:image-tag
```
Once the container is running,  go to  [localhost:80](http://localhost:80) to see the if the server is working or not. If it is running a message will be displayed

## Making Server Requests
The `POST` method is used to make requests for predictions. Every request should contain the data that represents a wine in `JSON` format like below :

### For no-batch
```
{
  "alcohol":12.6,
  "malic_acid":1.34,
  "ash":1.9,
  "alcalinity_of_ash":18.5,
  "magnesium":88.0,
  "total_phenols":1.45,
  "flavanoids":1.36,
  "nonflavanoid_phenols":0.29,
  "proanthocyanins":1.35,
  "color_intensity":2.45,
  "hue":1.04,
  "od280_od315_of_diluted_wines":2.77,
  "proline":562.0
}
```

Similarly, the `curl` can be used to make predictions using a `JSON` file. The no-batch folder contains and `wine-examples` folder which contains a few `JSON` files that can be used for requesting predictions. This can be done using the following command in the command line:
```
curl -X POST http://localhost:80/predict \
    -d @./wine-examples/1.json \
    -H "Content-Type: application/json"
```

**Note :** The directory has to be changed to the `no-batch` folder before the curl command can be run


### For With-Batch
As is the case for `no-batch`, predictions for `with-batch` can be made using `curl`. For example, the following curl command can be used for batch prediction :
```
curl -X POST http://localhost:81/predict \
    -d @./wine-examples/batch_1.json \
    -H "Content-Type: application/json"
```
**Note :** The directory has to be changed to the `with-batch` folder before the curl command can be run
