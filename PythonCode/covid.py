from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

from google.colab import drive
drive.mount('/content/drive')

Classify = Sequential();

Classify.add(Conv2D(64,(3,3),input_shape=(64,64,3),activation='relu'))
Classify.add(MaxPooling2D(pool_size=(2,2)))

Classify.add(Conv2D(32,(3,3),activation='relu'))
Classify.add(MaxPooling2D(pool_size=(2,2)))

Classify.add(Flatten())
Classify.add(Dense(units=104,activation='relu'))
Classify.add(Dense(units=1,activation='sigmoid'))


Classify.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255,shear_range=0.4,zoom_range=0.3,horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)


new_train = train_data.flow_from_directory('/content/drive/MyDrive/dataaasettttt/newdataset/Covid19-dataset/train',target_size=(64,64),batch_size=4,class_mode='binary')
new_test = train_data.flow_from_directory('/content/drive/MyDrive/dataaasettttt/newdataset/Covid19-dataset/test',target_size=(64,64),batch_size=4,class_mode='binary')

# y_prediction = Classify.predict(new_test)
# result = confusion_matrix(new_test, y_prediction , normalize='pred')
# print(result)



Classify.fit_generator(new_train,steps_per_epoch=40,epochs=5,validation_data=new_test,validation_steps=8)



import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

# Predict on test data
Y_pred = Classify.predict_generator(new_test)
y_pred = np.round(Y_pred)


print(Y_pred)


# Get true labels from test data generator
y_true = new_test.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=new_test.class_indices.keys(),
            yticklabels=new_test.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


Accuracy = metrics.accuracy_score(y_true, y_pred)
print("This is accuracy " , Accuracy)

Precision = metrics.precision_score(y_true, y_pred)
print("The precision is ", Precision)



import numpy as np
from keras.preprocessing import image
test_img = image.load_img(r'/content/drive/MyDrive/newdataset/Covid19-dataset/train/Normal/01.jpeg',target_size=(64,64))
test_img = image.img_to_array(test_img)
#this is used to add the extra dimensions to the image before predicting to the image
test_img = np.expand_dims(test_img,axis=0)
res = Classify.predict(test_img)
# print(res)
new_test.class_indices
# print(new_test.class_indices)
# print(res) 
if res[0][0]==1:
  prediction='Normal'
  print(prediction)
else:
  prediction='Covid'
  print(prediction)
