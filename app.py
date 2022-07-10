#First we import flask, requests to make request on the API & jsonify to convert data into json
from flask import Flask, jsonify, request
#here we will use classifier function & to do that we need to import get_prediction function from classifer file
from classifier import  get_prediction
#Naming the app to the flask constructor soit knows where to look for the dependent files
app = Flask(__name__)


#Here we start writing the code to figure out the route 
#Next we need a post request to send the image to the prediction modek & our route name will be 'predict-digit'
@app.route("/predict-digit", methods=["POST"])

#Inside this route we have a function predict_data. We will use the request function to get files from the digit key
#We'll also use the get_prediction function here & written the prediction in json format with status code of 200
def predict_data():
  # image = cv2.imdecode(np.fromstring(request.files.get("digit").read(), np.uint8), cv2.IMREAD_UNCHANGED)
  image = request.files.get("digit")
  prediction = get_prediction(image)
  return jsonify({
    "prediction": prediction
  }), 200

#We use this condition block to prevent any other code from executing simultaneously while the model is being imported
#We write debug = true so that the server will update every time you save the code
if __name__ == "__main__":
  app.run(debug=True)
  
  
#1.Run this code & copy the port on which the code is running
#2.Test the API in the post man
#3.Make a post request on the route 
#4.We have to make the request in the "form-data" as we will be giving image files to the API
#5.In the place of "key" we have to write "digit" & type as file. As it's value we'll be using digit image as file & make a request
#to see the prediciton

#why we created a classification model in a separate file instead of doing it in the api file itself?
#The Classifier takes time to learn from the training data that we provide. If we build it in the API itself, then it will take a lot of time to give the response.
#why did we make a separate function to process the image and predict the outcome, instead of doing it in the API?
#Flask APIs, to optimize the performance speed, caches the APIs so that whenever an API call is made, it can return the data much faster. If we did this in the API itself, it will return the same output for different images that we send to the server.
#What we did today is known as the MVC Architecture. MVC stands for Model View Controller .There are also other types of architectures like MVVM,MVP, MVA etc. Keeping in mind that the API caches the data so that it can give a response faster, it is ideal to keep your processing code separately from your API.
#Model is generally referred to the Database, if any. Here, we are not using any Databases. View is referred to the API, which accepts a request and returns a response. Controller is the part that does all the heavy work, function(e){return a}, data processing, building classifier, etc. Hence, it is known as the MVC Architecture!