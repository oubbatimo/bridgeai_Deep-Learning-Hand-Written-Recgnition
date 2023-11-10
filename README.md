<img width="258" alt="image" src="https://github.com/oubbatimo/bridgeai_LLMs/assets/92709052/82b3e818-d545-46e5-8eb8-b328bf73b65f">

# Deep Learning for Handwritten Digit Recognition
Training uses MINST Dataset taken from Keras framework. Training and Test Data consist of 60000 and 10000 images respectively, in which all images have same size (28 by 28 pixels).

The neural network architecture:
Input Layer=28x28 Units, Hidden Layer (Dense)= 100 Units, Output Layer= 10 Units to decode every digit.



![image](https://github.com/oubbatimo/bridgeai_DigitRecognition/assets/92709052/1a2dd9eb-6951-484f-82f0-91d7854a19e7)





After training, I tested this network and had these results:

1. Llama 2 model with 7B parameters is already hosted on Replicated and we will access this model models through API.
2. You need to register on replicate and generate an API-Key.
3. We use Streamlit to create an interactive app for the Large Language Model Llama.
  


 ![Figure_1](https://github.com/oubbatimo/bridgeai_DigitRecognition/assets/92709052/d81fe4a9-2d85-493f-a209-d12cce71088f)

Prediction:6

Prediction=0 ![Figure_2](https://github.com/oubbatimo/bridgeai_DigitRecognition/assets/92709052/405bd059-71c9-4ac7-b7dc-9bb496a753c7)

Prediction=6 ![Figure_3](https://github.com/oubbatimo/bridgeai_DigitRecognition/assets/92709052/b7de6652-3a79-45e5-aced-a171631c259b)
