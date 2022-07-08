from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import os
import csv
import argparse
pd.options.mode.chained_assignment = None

class LogisticRegressionModel():
    def __init__(self):
        print('Initiating Program')

        self.args = self.get_args()

        kmers = product(['A','T','C','G'], repeat = 3)
        kmers = [kmer for kmer in kmers]
        self.kmers = [''.join([i for i in kmer]) for kmer in kmers]

        ###############################
        ### Create the GC Intervals ###
        ###############################
        self.gc_lst = np.linspace(0.2,0.8,61)
        self.gc_intervals = []
        for index,item in enumerate(self.gc_lst):
            try:
                self.gc_intervals.append('%s-%s' % (round(self.gc_lst[index],3),round(self.gc_lst[index+1],3)))
            except:
                break

        #########################################################
        ### Analyze data and create logistic regression model ###
        #########################################################
        self.run_model()

    def get_args(self):

        print('Getting Args')
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--directory", help="Directory containing sample data",required=True)
        parser.add_argument("-s", "--samples", help="List of samples to analyze",required=True)
        parser.add_argument("-r", "--results", help="Results directory",required=True)
        args = parser.parse_args()

        self.sample_dict = {}
        with open(args.samples,'r') as mydata:
            for line in mydata:
                sample = line.split(' ')[0].strip()
                cancer_status = line.split(' ')[1].strip()
                self.sample_dict[sample] = cancer_status

        return args

    def run_model(self):
        print('Running Model')

        ###############################
        ### Compile the sample data ###
        ###############################
        self.endseq_features = {}
        self.mendseq_features = {}
        analyzed_samples = []
        for sample in self.sample_dict.keys():
            fh = os.path.join(self.args.directory,'%s.mendseq.txt' % sample)
            check = self.analyze_data(sample,fh)
            if check is None:
                continue
            else:
                analyzed_samples.append(sample)

        #############################################################
        ### Build a numpy array for the logistic regression model ###
        #############################################################
        endseq_data = []
        mendseq_data = []
        y = []
        for sample in analyzed_samples:
            status = self.sample_dict[sample]
            if status == 'Cancer':
                y.append(1)
            elif status == 'Healthy':
                y.append(0)
            endseq_data.append([self.endseq_features[sample][kmer] for kmer in self.kmers])
            iter = []
            for kmer in self.kmers:
                iter.extend(self.mendseq_features[sample][kmer])
            mendseq_data.append(iter)
        print(mendseq_data)
        ####################
        ### EndSeq Model ###
        ####################
        endseq_tprs, endseq_aucs, endseq_scores = self.logistic_regression(np.array(endseq_data),np.array(y),analyzed_samples)

        #####################
        ### MendSeq Model ###
        #####################
        mendseq_tprs, mendseq_aucs, mendseq_scores = self.logistic_regression(np.array(mendseq_data),np.array(y),analyzed_samples)

    def analyze_data(self,sample,fh):

        #################################################
        ### Make sure the file for this sample exists ###
        #################################################
        if not os.path.exists(fh):
            print('%s does not exist' % fh)
            return None
        else:
            self.endseq_features[sample] = {}
            self.mendseq_features[sample] = {}

        #####################################
        ### Read the data for this sample ###
        #####################################
        df = pd.read_csv(fh,sep='\t')
        total_fragments = np.sum(df['Count'])
        df = df[df['GC Bin'].isin(self.gc_intervals)]

        ################################################
        ### Calculate P(GC) for use in bayes theorem ###
        ################################################
        gc_prob = []
        for gc in df['GC Bin'].unique():
            temp = df[df['GC Bin'] == gc]
            gc_prob.append(np.sum(temp['Count'])/total_fragments)

        ##################################################
        ### For each kmer, calculate the conditional   ###
        ### probaility, P(Kmer|GC) using Bayes Theorem ###
        ##################################################
        for kmer in df['3mer'].unique():
            temp = df[df['3mer'] == kmer]

            ######################################
            ### calculate the EndSeq Frequency ###
            ######################################
            self.endseq_features[sample][kmer] = np.sum(temp['Count'])/total_fragments
            tot = np.sum(temp['Count'])

            #######################################
            ### calculate the MendSeq Frequency ###
            #######################################
            temp['Conditional'] = temp['Count']/tot
            temp['Joint'] = temp['Conditional']*self.endseq_features[sample][kmer]
            lst = []
            for joint,gc in list(zip(temp['Joint'].tolist(),gc_prob)):
                lst.append(float(joint)/gc)
            temp['Frequency'] = lst
            self.mendseq_features[sample][kmer] = temp['Frequency'].tolist()

        return True

    def logistic_regression(self,X,y,samples):

        mean_fpr = np.linspace(0, 1, 100)
        repeats = 10
        scores = {}
        tprs = []
        aucs = []
        for repeat in range(repeats):
            scores[repeat] = {}

            ########################################
            ### Create a cross-validation object ###
            ########################################
            cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
            cycles = [iter for iter in enumerate(cv.split(X,y))]

            ################################
            ### Perform Cross-Validation ###
            ################################
            for cycle,(train,test) in cycles:
                X_train = X[train,:]
                for item in X_train:
                    item[np.isnan(item)] = 0
                X_test = X[test,:]
                for item in X_test:
                    item[np.isnan(item)] = 0
                y_train = y[train]
                y_test = y[test]
                test_samples = []
                for idx in test:
                    test_samples.append(samples[idx])
                train_samples = []
                for idx in train:
                    train_samples.append(samples[idx])

                ##################################################
                ### Fit a Scaler to data from the Training Set ###
                ##################################################
                scaler = StandardScaler().fit(X_train)

                ########################################################
                ### Scale the data to a standard normal distribution ###
                ########################################################
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                ####################################################
                ### Fit the Regression Model to the Training Set ###
                ####################################################
                ols = LogisticRegression()
                with ignore_warnings(category=ConvergenceWarning):
                    ols.fit(X_train,y_train)

                ##########################
                ### Score the Test Set ###
                ##########################
                y_pred = ols.predict_proba(X_test)
                for count,item in enumerate(y_pred[:,1]):
                    sample = test_samples[count]
                    scores[repeat][sample] = item

                #############################################
                ### Evaluate the Performance of the Model ###
                #############################################
                auc = roc_auc_score(y_test, y_pred[:,1].reshape(-1,1))
                aucs.append(auc)
                print('AUC: %.3f' % auc)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1].reshape(-1,1))
                fpr_tpr = list(zip(fpr, tpr))
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                tprs.append(interp_tpr)

        return tprs, aucs, scores

if __name__ == '__main__': LogisticRegressionModel()
