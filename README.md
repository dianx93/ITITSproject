# ITITSproject

##Setup

To run this, you need to have OpenCV 3.2.0+ and scikit-learn installed.

Next, go to http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads and download Training set's and Test set's images and HOG features.
And also http://btsd.ethz.ch/shareddata/  and download BelgiumTSC_Training

Put training data into "GTSRB/Final_Training/Images/" and "BelgianTS/Training/". Create the root level folders if not exsisting.

Run the program to be read in.

On first run in file classificator.py comment out line 64 and remove comment signs from lines 60-62. One first run is done 60-62 ca nbe commented out and 64 commented int to improve speed. Big part of waiting time comes from preprocessing the training data.
