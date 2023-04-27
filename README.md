# CNN-Small-The-optimal-CNN-network-structure-for-small-sized-images.
 CNN-Small architecture has been proven to be effective in handling small-sized and low-resolution image datasets with an accuracy of up to 95% for increasing image sizes from 32 x 32 px to 128 x 128 px.. Its multi-layer convolutional design efficiently extracts image features while reducing the number of parameters and accelerating the model's computation speed. By utilizing dropout and L1/L2 regularization algorithms, the model can effectively eliminate overfitting and improve generalization capabilities. The use of 2D convolutional layers with a 3x3 kernel and max pooling layers with a 2x2 size ensures that important features are retained while irrelevant ones are removed, further enhancing the model's ability to recognize and classify images. In summary, the CNN-Small architecture has been proven effective in handling small-sized and low-resolution image datasets, making it a suitable choice for image recognition tasks in these domains.
 <img link="illustration_structure.png">
 The CNN-Small model was used to train on a dataset of diseased rice images, using two different image characteristics: a standard image set with a size of 320 x 320 pixels, and a mini image set with a size of only 32 x 32 pixels. This allows the model to learn how to classify and detect rice diseases on both high and low-resolution images.

 The dataset used to train the model includes thousands of diseased and non-diseased rice images, with an accuracy rate of over 95%. By using the CNN-Small model, we can classify diseased rice images quickly and accurately, helping farmers and rice researchers detect and treat rice diseases in a timely manner.

 An illustration of the training dataset is shown in the figure below.
 <img link="sample_dataset_images.png">
Contributors:
Quach Luyl Da, FPT University, VietNam
Nguyen Quoc Khang, FPT University, VietNam
Nguyen Quynh Anh, FPT University, VietNam
Tran Ngoc Hoang, FPT University, VietNam 
Paper: Updating