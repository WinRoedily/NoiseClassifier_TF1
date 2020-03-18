# NoiseClassifier_TF1
Train extracted MFCCs and labels saved in .h5 files using the CNN-LSTM method and TensorFlow 1.15 rc3 library. The program converts the model into a TensorFlow Lite model.

This is just a simple Python code that will read both train and test dataset saved in .h5 files. The TFLite_LSTM_Experiment class utilizes h5py and TensorFlow 1.15.0rc3 libraries.
## Here's how it works:
1. The h5py will read the dataset in both .h5 files, looking for datasets called **data** and **label**.
2. I convert the label to categorical utilizing Keras and determine the number of epochs afterwards.
3. This model is summarized as:
  * Input (*number of frames* x *MFCCs amount*)
  * Conv1D (filters=64)
  * MaxPooling1D (pool_size=4)
  * Conv1D (filters=128)
  * MaxPooling1D (pool_size=4)
  * StackedRNNCells:
    * TFLiteLSTMCell (num_units=64)
    * TFLiteLSTMCell (num_units=32)
    * DynamicRNN
    * Unstack
  * Fully Connected (units=5) with *Softmax* activation
  * Compile with *Adam* optimizer and *binary_crossentropy* loss function
4. After the training and validation is complete, the program converts the sessions into TensorFlow Lite model. This model can run very well on Android smartphone where the application is available on my other repository called **Mobile_Noise_Classifier**.

I hope this simple code can be useful, and may God bless you. :angel:
