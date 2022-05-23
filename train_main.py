from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from classifier import training
from preprocess import preprocesses

datadir = './aligned_img'
input_datadir = './train_img'
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'

"""[===== Data Preprocesses =====]"""
obj = preprocesses(input_datadir, datadir)
nrof_images_total, nrof_successfully_aligned = obj.collect_data
print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

"""[===== Data Training =====]"""
print("Training Start")
obj = training(input_datadir, modeldir, classifier_filename)
get_file = obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")
