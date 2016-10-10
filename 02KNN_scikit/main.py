import cPickle
import os
from random import random
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from class_Passage import Passage, create_passage_obj_list
from class_AllPassages import AllPassages
from class_Dataset import Dataset
from ModelSelectionMetrics import MultiLabelClassifierMetricsCalculator

from settings import Pickle_Dir

# def main():
if not ( os.path.isfile(Pickle_Dir + "TrainingDataset.pickle") \
            and os.path.isfile(Pickle_Dir + "TestingDataset.pickle") \
            and os.path.isfile(Pickle_Dir + "ErrorAnalysisDataset.pickle") ):
    # load & parsing & preprocessing
    # -> create a list of Passage object
    psg_list = create_passage_obj_list()
    print  "[Main] Passage List of %d passages got." % len(psg_list)
    # Create AllPassage object -> generate feature_vector_attr_list
    all_psg = AllPassages(psg_list)
    feature_vector_attrs = all_psg.gen_feature_vector_attrs()
    # Create data set
    training_ds  = Dataset(feature_vector_attrs)
    testing_ds  = Dataset(feature_vector_attrs)
    error_analysis_ds  = Dataset(feature_vector_attrs)
    # calculate sample's Feature Vector(TF-IDF) -> add into dataset
    for psg in psg_list:
        fv = psg.cal_TF_IDF(feature_vector_attrs, all_psg.IDF)
        rand_val = random()
        if rand_val < 0.7:  # Training Dataset
            training_ds.add(psg.new_id, psg.topics, fv)
        elif rand_val < 0.9:  # Testing Dataset
            testing_ds.add(psg.new_id, psg.topics, fv)
        else: # Error Analysis Group
            error_analysis_ds.add(psg.new_id, psg.topics, fv)
    # dump data_set into cPickle
    with open("data_pickles/TrainingDataset.pickle" , "w" ) as fh:
        cPickle.dump( training_ds, fh)
    with open("data_pickles/TestingDataset.pickle" , "w" ) as fh:
        cPickle.dump( testing_ds, fh)
    with open("data_pickles/ErrorAnalysisDataset.pickle" , "w" ) as fh:
        cPickle.dump( error_analysis_ds, fh)
    for psg in psg_list:
        del psg
    del psg_list, all_psg

# load datasets from cPickle
training_ds = cPickle.load( open(Pickle_Dir + "TrainingDataset.pickle", "r") )
testing_ds = cPickle.load( open(Pickle_Dir + "TestingDataset.pickle", "r") )
error_analysis_ds = cPickle.load( open(Pickle_Dir + "ErrorAnalysisDataset.pickle", "r") )

if not ( os.path.isfile(Pickle_Dir + "KNN_Classifier_MyImplementation.pickle") \
            and os.path.isfile(Pickle_Dir + "KNN_MLB.pickle") ):
    # KNN Classification,  http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='kd_tree', metric ="minkowski", p=2)
    # 0.76  with min_samples_leaf=10, min_weight_fraction_leaf=0.2, max_depth= 500
    # 0.76  with min_samples_leaf=10, min_weight_fraction_leaf=0.2, max_depth=500
    # 0.69  with min_samples_leaf=30, min_weight_fraction_leaf=0.2, max_depth=500
    # 0.79  with min_samples_leaf=5, min_weight_fraction_leaf=0.2, max_depth=500
    # 0.79  wtih min_samples_leaf=5, min_weight_fraction_leaf=0.2, max_depth=1000

    # Preprocessing - MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=training_ds.classes_, sparse_output=False)
    training_ds.class_label_lists_fitted = mlb.fit_transform(training_ds.class_label_lists)
    print training_ds.size, testing_ds.size, error_analysis_ds.size
    # Model Training
    t0 = time()
    clf.fit(training_ds.fv_nparray, training_ds.class_label_lists_fitted)
    training_time = time() - t0
    print "[main.py] Offline cost for training from %d samples is %.04f seconds." \
                % (len(training_ds.fv_nparray), training_time)
    # Dump Model into pickle
    with open( Pickle_Dir +  "KNN_Classifier_MyImplementation.pickle" , "w" ) as fh:
        cPickle.dump( clf, fh)
    with open( Pickle_Dir +  "KNN_MLB.pickle" , "w" ) as fh:
        cPickle.dump( mlb, fh)

clf = cPickle.load( open( Pickle_Dir +  "KNN_Classifier_MyImplementation.pickle" , "r" ) )
mlb = cPickle.load( open( Pickle_Dir +  "KNN_MLB.pickle" , "r" ) )

# Accuracy & Efficient Measurement
metrics = MultiLabelClassifierMetricsCalculator()
t0 = time()
for i in xrange(len(testing_ds.fv_nparray)):
    prediction = mlb.inverse_transform(clf.predict(testing_ds.fv_nparray[ i , : ].reshape(1, -1)))[0]
    metrics.addSample(testing_ds.class_label_lists[i], prediction)
delta_t = t0 - time()
print "[main.py] Time Taken: %.02f seconds on %d number of testing samples." % (delta_t, len(testing_ds.fv_nparray))
print metrics


# # Accuracy Measurement
# total_num = testing_ds.size
# correct_num = 0
# for idx in xrange(total_num):
#     sample_fv = testing_ds.fv_nparray[idx, :].reshape(1, -1)
#     labels = testing_ds.class_label_lists[idx]
#     prediction = clf.predict(sample_fv)  # output an binary array
#     # filtering the labels from prediction
#     # predicted_labels = map( lambda i: mlb.classes[i], filter( lambda i : prediction[0][i] == 1, range(len(prediction[0]))))
#     predicted_labels = mlb.inverse_transform(prediction)
#     for pred in predicted_labels[0]:
#         if pred in labels:
#             correct_num += 1
#             # print pred, labels, predicted_labels
#             break
# print "[Accuracy] Total: %d, Correct: %d, Percentage: %.02f" % (total_num, correct_num, float(correct_num)/ total_num)
