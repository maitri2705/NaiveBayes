import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.stem import PorterStemmer
import sys
ps = PorterStemmer()

def openDirectory(dirName):
    os.chdir(dirName)

class Process:
    stop_words = set(stopwords.words('english'))
    directories = ['comp.graphics/','misc.forsale/','rec.autos/','rec.motorcycles/','sci.space/']
    commonDirectoryPath1=sys.argv[1]
    commonDirectoryPath2=sys.argv[2]
   # commonDirectoryPath1 = 'C:/Users/MAITRI SHAH/Desktop/Studies/Utd/Sem1/ML/Assignment4/20news-bydate/20news-bydate-train/'
   # commonDirectoryPath2='C:/Users/MAITRI SHAH/Desktop/Studies/Utd/Sem1/ML/Assignment4/20news-bydate/20news-bydate-test/'
    def readAllFiles(self):
        uniqueWordsList1 = []
        classLabels1=[]
        for i in self.directories:
            pathForOsWalk = ''.join([self.commonDirectoryPath1,i])
           # print(pathForOsWalk)
            os.chdir(pathForOsWalk)
            #print(pathForOsWalk, "reading has been started..... ")
            j = 1;
            for dirpath, dirname, files in os.walk(pathForOsWalk):
                for file in files:
                    uniqueWordsList1.append(self.readSingleFile(file))
                    classLabels1.append(i)
                    j = j + 1
            #print(pathForOsWalk, "reading finished..... ",i,"Total filles read")
        #print(uniqueWordsList1)
        uniqueWordsList2 = []
        classLabels2 = []
        for i in self.directories:
            pathForOsWalk = ''.join([self.commonDirectoryPath2, i])
            #print(pathForOsWalk)
            os.chdir(pathForOsWalk)
           # print(pathForOsWalk, "reading has been started..... ")
            j = 1;
            for dirpath, dirname, files in os.walk(pathForOsWalk):
                for file in files:
                    uniqueWordsList2.append(self.readSingleFile(file))
                    classLabels2.append(i)
                    j = j + 1
            #print(pathForOsWalk, "reading finished..... ", i, "Total filles read")
        return uniqueWordsList1,classLabels1,uniqueWordsList2,classLabels2

    def readSingleFile(self,file):
        f = open(file,'r')
        uniqueWords = []
        filteredSentence = []
        finished = False
        for line in f:
            if line.startswith('Lines:') or line.startswith('Subject:') or line.startswith('Organization:'):
                finished = True
            if (line.startswith('From:') or line.startswith('Article-I.D.:') or line.startswith('Expires:') or line.startswith('Reply-To:')):
                finished = False
            if finished:
                words = word_tokenize(line)
                tokens = (e for e in words if e.isalpha())
                for w in tokens:
                    if w not in self.stop_words:
                        w=ps.stem(w)
                        filteredSentence.append(w)
                        if w not in uniqueWords:
                            uniqueWords.append(w)

        f.close()
        return uniqueWords

    def theNaiveBayes(self,uniqueList1,classes1,uniqueList2,classes2):
        trainInstances=uniqueList1
        classLabels=classes1
        testInstances=uniqueList2
        testLabel=classes2

        def getUniqueFeatures(testInstance):
            return list(set(testInstance))

        def getClassLabels(classLabels):
            return list(set(classLabels))

        def getDictionarySize(trainInstances):
            uniqueLabelList = []
            for i in range(len(trainInstances)):
                uniqueLabelList = uniqueLabelList + trainInstances[i]

            uniqueLabelList = list(set(uniqueLabelList))
            size = (len(uniqueLabelList))
            return size

        def getCountOfFeatureInClass(featureValue, classLabel, trainInstances, classLabels):
            count = 0
            for i in range(len(classLabels)):
                if (classLabels[i] == classLabel):
                    count = count + trainInstances[i].count(featureValue)
            return count

        def getTotalFeaturesInClass(classLabel, trainInstances, classLabels):
            count = 0
            for i in range(len(classLabels)):
                if (classLabels[i] == classLabel):
                    count = count + len(trainInstances[i])
            return count

        def getPrior(classLabel, classLabels):
            total = len(classLabels)
            count = classLabels.count(classLabel)
            return float(count / total)

        uniqueFeatures = getUniqueFeatures(testInstances[0])

        uniqueClassLabels = getClassLabels(classLabels)

        conditionalProb = np.zeros((len(uniqueFeatures), len(uniqueClassLabels)))

        priorProb = np.zeros((len(uniqueClassLabels)))

        for i in range(len(uniqueClassLabels)):
            priorProb[i] = float(getPrior(uniqueClassLabels[i], classLabels))

        B = getDictionarySize(trainInstances)

        for i in range(len(uniqueFeatures)):
            for j in range(len(uniqueClassLabels)):
                conditionalProb[i][j] = (getCountOfFeatureInClass(uniqueFeatures[i], uniqueClassLabels[j],trainInstances, classLabels) + 1) / (getTotalFeaturesInClass(uniqueClassLabels[j], trainInstances, classLabels) + B)
        listProbResults = np.zeros((len(testInstances), len(uniqueClassLabels)))

        for m in range(len(testInstances)):
            for i in range(len(uniqueClassLabels)):
                listProbResults[m][i] = priorProb[i]
                for j in range(len(testInstances[m])):
                    if(uniqueFeatures.__contains__(testInstances[m][j])):
                       # if(conditionalProb.__contains__(uniqueFeatures.index(testInstances[m][j]))):
                        listProbResults[m][i] = listProbResults[m][i] * conditionalProb[uniqueFeatures.index(testInstances[m][j])][i]
        #print(listProbResults[0][0]," ",len(testLabel)," ",len(uniqueClassLabels))
        counter = 0
        for j in range(len(listProbResults)):
            max = -1
            maxNumber=-1
            for i in range(len(uniqueClassLabels)):
                temp = listProbResults[j][i]
                if (temp > max):
                    max = temp
                    maxNumber = i
                # if(j==10):
                #     #print(i," ",listProbResults[j][i])
            tempAns = uniqueClassLabels[maxNumber]
            # if(j==10):
            #     print(tempAns," ",testLabel[j])
            if (tempAns == testLabel[j]):
                counter = counter + 1
        accuracy = counter / (len(testLabel))
        return accuracy

''' ========================  MAIN FUNCTION ============================='''
if __name__ == "__main__":
    pp = Process()
    uniqueWordList1,classLabel1,uniqueWordList2,classLabel2=pp.readAllFiles()
    accu=pp.theNaiveBayes(uniqueWordList1,classLabel1,uniqueWordList2,classLabel2)
    print("Accuracy",(accu*100))