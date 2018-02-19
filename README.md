#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

Here is a link to my [project code]()

###Data Set Summary & Exploration

####1. Basic summary of the data set

Native Python methods were used to retrieve the number of samples and their size::

* Size of training = 34799
* Size of the validation set = 4410
* Size of test set = 12630
* Shape of traffic sign image is (32, 32, 3), Heigh = 32, width = 32 and 3 channels for R, G, B encoded in 8 bits (0 to 255)
* The size of unique classes = 43

####2. Include an exploratory visualization of the dataset.

A grid (matplotlib.gridspec) was used to displayed both color and gray scaled images, with 16 items per row,
and 5 rows. The following shortcut was used to get grayscaled images from the original RGB:

gray_scale_image = np.dot(X_test[i][...,:3], [0.333, 0.333, 0.333])
which has a quite good performance.

![colorimages.png](./examples/colorimages.png "Color images")

![grayscaleimages.png](./examples/grayscaleimages.png "Grayscale images")


One histogram for each labels data (training, validation and testing) was rendered:
![traininhist.png](./examples/traininhist.png "Training histogram")
![validationhist.png](./examples/validationhist.png "Validation histogram")
![testinghist.png](./examples/testinghist.png "Testing histogram")

###Design and Test a Model Architecture

####1. Description of image data preproces.

The following simple function was used to:
 a) Turn images to grayscale
 b) normalize values deom 0 - 255 range to -1, 1

 def normalize_image(X):
    X_gs = np.zeros((32, 32, 1))

    for x in range(0, 31):
        for y in range(0, 31):
            r = (float(X[x, y, 0]) - 128.000) / 128.000 * 0.333
            g = (float(X[x, y, 1]) - 128.000) / 128.000 * 0.333
            b = (float(X[x, y, 2]) - 128.000) / 128.000 * 0.333
            X_gs[x, y, 0] = r + g + b
    return X_gs

The function is not fast, but reasonable for small samples, and works for this project.
In realtime more OpenCV or similar oriented functions should be used, or a C++ version which should
be n - times faster.

####2. Model Description:

The model is based on a LeNet CNN, with droup out,
and it is as follows:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|Flatten 		|	Input = 5x5x16. Output = 400			|
|Fully Connected			|	Input = 400, Output = 120 |
| RELU					|												|
| Dropout					|	0.80					|
|Fully Connected			|	Input = 120, Output = 84 |
| RELU					|												|
| Dropout					|	0.80					|
|Fully Connected			|	Input = 84, Output = 43 |

All weightswere initialized with a truncated normal with mean = 0 and stddev = 0.1

####3. Description of model training process.

After some testing, and with a simple i3 processor (no GPU at all),
the following hyperparameters were selected:

epochs = 24
batch_size = 1024 #bigger batches produced more accurate predictions
rate = 0.005 #Training rate

An Adam optimizer was used and the implementation of cross entropy and optimizer is quite simple:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

During training, accuracy values are dsplayed for each epoch,
to provide a feedback to fine tune the model.

Finally the session is persisted in lu_covnet file
(but for some reason when restored, resulting predictions were not correct,
so I decided, after some trouble shooting, to re train it for the addicional
files from the web).

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
Training Accuracy = 0.998
Validation Accuracy = 0.928
Testing Accuracy = 0.924

Accuracy note: I noticed that when increasing the epochs number
the model was overfiting and becoming wors when generalizing
with new samples. hence I decided to keep accuracy values relative low
due to that reason.

After running the model for the first time with smaller batch size (128),
no more than 10 epochs, a lower training rate (0.001) and without dropout,
accuracy was low, and for web downloaded images, it didn't work at all.
I added dropout, increased the batch size to 512 and epochs to 16, and
results were much better. But as I have no GPU in my system, I tried
reducing the learning rate to 0.05, but increased batch to 1024 and epochs
to 24 and results were far better.

For the sake of a 'safe path' I based my self on the old LeNet
architecture, with batch date due to system memory constrains,
as this was my very first practical project of a CovNet.
I also selected this schema after doing someo research about image clasification
with CovNets, so I remained conservative (just added the dropout to get better results).
Overall results proved that this model is accurate (and for sure can be more accurate
with more fine tunning, more epochs for example and a lower training rate, but
that can be tested with a GPU enabled system).

###Test a Model on New Images

####1. I selected 6 german images
Here are five German traffic signs that I found on the web:

![11_right_of_way.png](./samples_from_web/11_right_of_way.png "Right Of Way")
![13_yield.png](./samples_from_web/13_yield.png "Yield")
![1_speed_limit_30.png](./samples_from_web/1_speed_limit_30.png "Speed Limit 30")
![22_bumpyroad.png](./samples_from_web/22_bumpyroad.png "Bumpy Road")
![4_speed_limit_70.png](./samples_from_web/4_speed_limit_70.png "Speed Limit 70")
![4_speed_l7_speed_limit_100.png](./samples_from_web/7_speed_limit_100.png "Speed Limit 100")

In particular Speed Limit 30 and 100 is in perspective hence hard to clasify for the model.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed Limit 30		| Speed Limit 30								|
| Speed Limit 70	    | Speed limit (20km/h)			 				|
| Speed Limit 100		| No passing for vehicles over 3.5 metric tons	|
| Right Of Way      	| Right-of-way at the next intersection 		|
| Yield     			| Yield 										|
| Bumpy Road			| Bumpy Road									|


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.66%.
This is reasonable because of the quality of the images not correctly predicted.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is 100% sure that this is a speed limit 30 (probability of 100%)

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Speed limit (30km/h)							|
| .0     				| Speed limit (50km/h)							|
| .0					| Roundabout mandatory							|
| .0	      			| Speed limit (100km/h) 		 				|
| .0				    | Speed limit (80km/h) 							|
| .0				    | Speed limit (60km/h)							|


For the second image the model predicts wrong with a 90% a Speed limit (20km/h)
for a expected Speed limit 70 sign, and the second prediction is the correct one

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .90         			| Speed limit (20km/h)							|
| .09     				| Speed limit (70km/h)							|
| .01					| Speed limit (30km/h)							|
| .00	      			| Roundabout mandatory							|
| .00				    | Speed limit (100km/h)							|
| .00				    | Keep right 									|

For the third image the model predicts wrong with a 70% a No passing for vehicles over 3.5 metric tons
for a expected Speed limit 100 sign, and the third probability is for the expected sign
but with only a 6%

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .70         			| No passing for vehicles over 3.5 metric tons	|
| .23     				| Roundabout mandatory							|
| .06					| Speed limit (100km/h)							|
| .00	      			| Vehicles over 3.5 metric tons prohibited		|
| .00				    | No passing     								|
| .00				    | Go straight or left 							|

For the fourth image the model predicts correctly a Right Of Way with a 99% probability

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99        			| Right-of-way at the next intersection			|
| .00     				| General caution 								|
| .00					| Beware of ice/snow   							|
| .00	      			| Pedestrians 					 				|
| .00				    | End of no passing by vehicles over 3.5 metric tons|
| .00				    | Priority Road									|

For the image five the model predicts correctly a Yield a 100% probability

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0        			| Yield   										|
| .00     				| Keep right									|
| .00					| No passing									|
| .00	      			| Turn left ahead 					 			|
| .00				    | No passing for vehicles over 3.5 metric tons	|
| .00				    | Ahead only     								|

For the image six the model predicts correctly a Bumpy Road, with a 99%

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99        			| Bumpy Road	   								|
| .00    				| Children crossing								|
| .00					| Dangerous curve to the right					|
| .00	      			| No passing				 					|
| .00				    | Keep right									|
| .00				    | Yield     									|
