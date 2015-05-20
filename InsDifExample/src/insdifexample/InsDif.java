package mulan.classifier;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.matrix.Matrix;
/**
 *
 * @author  Agathangelou Sofia
 * @author  Liaros Georgios
 * @author  Paraskevas Eleftherios
 * @author  Tzanakas Alexandros
 * @version 2015.05.20
 * 
 */

public class InsDif extends MultiLabelLearnerBase implements MultiLabelLearner {
    
    private static Matrix matrixFai;
    Matrix weights;
    Matrix tVec;
    Matrix train;
    int[] clusterCenterIndices;
    Matrix normTrain;
    Matrix innerTrainCenter;
    Matrix innerCenterCenter;
    double[] normCenter;
    private float ratio;
    public InsDif(float clusterRatio){
        ratio = clusterRatio;
    }
    
    public InsDif(){
        ratio = 0.1f;
    }
    
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        
        // Initializations
        int numOfInstances = trainingSet.getNumInstances();
        numLabels = trainingSet.getNumLabels();
        ArrayList<ArrayList<double[]>> list = new ArrayList<>();
        int i, j;
        int numOfAttr = trainingSet.getDataSet().numAttributes();
        int numOfAttrWithoutLabels = numOfAttr - numLabels;
        double [][] listAttr = new double[numOfInstances][numOfAttrWithoutLabels];
        double [][] listLabels = new double[numOfInstances][numLabels];
        
        // Create the prototype vectors
        for(i=0; i< numOfInstances; i++) {
            for(j=0; j < numOfAttrWithoutLabels; j++) { //0-71
                listAttr[i][j] = trainingSet.getDataSet().get(i).value(j); // Create a list of attributes
            }
            for(j = numOfAttrWithoutLabels; j < numOfAttr; j++ ){ //72 - 77
                listLabels[i][j - (numOfAttrWithoutLabels)] = trainingSet.getDataSet().get(i).value(j);
            }
        }
        for(i = 0; i < listLabels[0].length; i++){
            ArrayList<double[]> instancesForClassI = new ArrayList<>();
            for(j = 0; j < numOfInstances; j++){
                if(listLabels[j][i] == 1){
                    instancesForClassI.add(listAttr[j]);
                }
            }
            list.add(instancesForClassI);    
        }

        // Calculate the mean vectors
        double sum[] = new double[numOfAttrWithoutLabels];
        double prototypeVectors[][] = new double[numLabels][numOfAttrWithoutLabels];
        int k;
        for(i = 0; i < list.size();i++) {
            sum = new double[numOfAttrWithoutLabels];
            for(j = 0; j < list.get(i).size(); j++) {
                for(k = 0; k < numOfAttrWithoutLabels; k++) {
                    sum[k] += list.get(i).get(j)[k]; // Sum it
                }
            }
            
            // Create the prototype vectors
            for(k = 0; k < numOfAttrWithoutLabels; k++) {
                prototypeVectors[i][k] = sum[k]/list.get(i).size();
            }
        }
        
        int numClusters = (int) (numOfInstances*ratio);
        train = new Matrix(listAttr);
        Matrix trainlabels = new Matrix(listLabels, numOfInstances, numLabels);
        normCenter = new double[numLabels];
        normTrain = new Matrix(1,numOfInstances); //1x391
        tVec = new Matrix(prototypeVectors); //6x72
        innerCenterCenter = new Matrix(numLabels,numLabels);// 6x6
        innerTrainCenter = new Matrix (numOfInstances, numLabels); //391x6
        Matrix innerTrainTrain = new Matrix (numOfInstances, numOfInstances); //391x391
        Matrix distanceMatrix = new Matrix(numOfInstances, numOfInstances);
        
        //calc norm_train
        Matrix s = train.arrayTimes(train).transpose();
        for(i = 0; i < s.getRowDimension(); i++){
            Matrix temp = s.getMatrix(i,i,0,s.getColumnDimension()-1);
            normTrain.plusEquals(temp);
        }
        
        //calc norm center
        for(i = 0; i < numLabels; i++){
            Matrix a = tVec.getMatrix(i,i,0,tVec.getColumnDimension()-1);
            Matrix b = tVec.getMatrix(i,i,0,tVec.getColumnDimension()-1).transpose();
            normCenter[i] = a.times(b).get(0,0);
        }

        //calc innerTrainTrain
        double temp;
        for(i = 0; i < numOfInstances; i++){
            innerTrainTrain.set(i, i, normTrain.get(0, i));
            for(j = i+1; j < numOfInstances;j++){
                Matrix a = train.getMatrix(i,i,0,train.getColumnDimension()-1);
                Matrix b = train.getMatrix(j,j,0,train.getColumnDimension()-1).transpose();
                temp = a.times(b).get(0,0);
                innerTrainTrain.set(i, j, temp);
                innerTrainTrain.set(j, i, temp);
            }
        }
        
        //calc innerTrainCenter
        for(i = 0; i < numOfInstances; i++){
            for(j = 0; j < numLabels; j++){
                Matrix a = train.getMatrix(i,i,0,train.getColumnDimension()-1);
                Matrix b = tVec.getMatrix(j,j,0, tVec.getColumnDimension()-1).transpose();
                innerTrainCenter.set(i,j,a.times(b).get(0,0));
            }
        }
        
        for(i = 0; i < numLabels; i++){
            for(j = 0 ; j < numLabels; j++){
                Matrix a = tVec.getMatrix(i, i, 0, tVec.getColumnDimension()-1);
                Matrix b = tVec.getMatrix(j, j, 0, tVec.getColumnDimension()-1).transpose();
                innerCenterCenter.set(i, j, a.times(b).get(0,0));
            }
        }
        
        for(i = 0; i < numOfInstances-1;i++){
            for(j= i+1; j < numOfInstances; j++){
                Matrix dist = new Matrix(numLabels, numLabels);
                for(int m = 0; m < numLabels; m++){
                    for(int n = 0; n < numLabels; n++){
                       double a,b,c,d,temp1,temp2,temp3;
                       a = normTrain.get(0, i);
                       b = normCenter[m];
                       c = innerTrainCenter.get(i,m);
                       temp1 = a+b-2*c;
                       a = normTrain.get(0,j);
                       b = normCenter[n];
                       c = innerTrainCenter.get(j,n);
                       temp2 = a+b-2*c;
                       a = innerTrainTrain.get(i,j);
                       b = innerTrainCenter.get(i,n);
                       c = innerTrainCenter.get(j,m);
                       d = innerCenterCenter.get(m,n);
                       temp3 = -2*(a-b-c+d);
                       dist.set(m, n, Math.sqrt(temp1+temp2+temp3));
                    }
                }
                distanceMatrix.set(i,j,getDist(dist));
            }
        }
        distanceMatrix.plusEquals(distanceMatrix.transpose());
        mIMLClustering(numClusters, distanceMatrix);
       
        Matrix inverseMatrixFai;
        inverseMatrixFai = matrixFai.inverse();
        weights = inverseMatrixFai.times(trainlabels);
    }
    
    private void mIMLClustering(int numCluster, Matrix distanceMatrix) {
        
        int numBags = distanceMatrix.getRowDimension();
        Matrix indicator = new Matrix(1, numBags, -1.0);
        Matrix newIndicator=new Matrix(1, numBags, -1.0);
        ArrayList<ArrayList<Integer>> clustering = new ArrayList<>(numCluster);
        ArrayList<ArrayList<Integer>> newClustering = new ArrayList<>(numCluster);
        int i;
        for(i = 0; i < numCluster; i++) {
            clustering.add(new ArrayList<>());
            newClustering.add(new ArrayList<>());
        }
        boolean success;
        int pointer;
        for(i = 0; i < numCluster; i++) {
            success = false;
            while(!success) {
                pointer = (int) Math.floor(numBags * Math.random());
                if(indicator.get(0, pointer) == -1) {
                    indicator.set(0, pointer, 1);
                    success = true;
                    clustering.get(i).add(pointer);
                }
            }
        }
        for(i = 0; i < numBags; i++) {
            if(indicator.get(0, i) == -1) {
                pointer = (int) Math.floor(numCluster * Math.random());
                clustering.get(pointer).add(i);
            }
        }
        
        indicator = new Matrix(1, numBags, -1.0);
        double curCenter[] = new double[numCluster];
        double newCurCenter[] = new double[numCluster];
        int clusterSize;
        Matrix temp;
        int j;
        int minIndex;
        for(i = 0; i < numCluster; i++) {
            clusterSize = clustering.get(i).size();
            temp = new Matrix(1, clusterSize, -1.0);
            for(j = 0; j < clusterSize; j++) {
                temp.set(0, j, sumMatrix(distanceMatrix, clustering.get(i), i));
            }
            minIndex = minimum(temp);
            int min = clustering.get(i).get(minIndex);
            clustering.get(i).clear();
            clustering.get(i).add(min); // Remove all but minimum
            indicator.set(0, min, i);
            curCenter[i] = min; 
        }
        success = false;
        int numIter = 0;
        int maxIter = 100;
        Matrix distance;
        int index;
        boolean noEmpty;
        int size;
        while(!success) {
            numIter++;
            if(numIter > maxIter) {
                break;
            }
            distance = new Matrix(numBags, numCluster, 0.0);
            for(i = 0; i < numBags; i++) {
                if(indicator.get(0, i) != -1) {
                    distance.setMatrix(i, i, 0, distance.getColumnDimension()-1, new Matrix(1,distance.getColumnDimension(),1.0));
                    distance.set(i, (int) indicator.get(0, i), -1.0);
                } else {
                    distance.setMatrix(i, i, 0, distance.getColumnDimension()-1, new Matrix(curCenter,1));
                }
            }
            for(i = 0; i < numBags; i++) {
                index = minimum(distance.getMatrix(i, i, 0, distance.getColumnDimension()-1));
                newClustering.get(index).add(i);
            }
            noEmpty = true;
            for(i = 0; i < numCluster; i++) {
                size = newClustering.get(i).size();
                if(size == 0) {
                    noEmpty = false;
                    break;
                }
            }
            boolean changed = false;
            if(noEmpty) {
                newIndicator=new Matrix(1, numBags, -1.0);
                newCurCenter = new double[numCluster];
                for(i = 0; i < numCluster; i++) {
                    int cluSize = newClustering.get(i).size();
                    temp = new Matrix(1, cluSize, -1.0);
                    for(j = 0; j < cluSize; j++) {
                        temp.set(0, j, sumMatrix(distanceMatrix, newClustering.get(i), i));
                    }
                    minIndex = minimum(temp);
                    int min = newClustering.get(i).get(minIndex);
                    newClustering.get(i).clear();
                    newClustering.get(i).add(min); // Remove all but minimum
                    newIndicator.set(0, min, i);
                    newCurCenter[i] = min;
                }
                Set<Double> set1 = new HashSet<>();
                Set<Double> set2 = new HashSet<>();
                
                for(i = 0; i < numCluster; i++) {
                    set1.add(curCenter[i]);
                    set2.add(newCurCenter[i]);
                }
                set1.removeAll(set2);
                if(set1.isEmpty()) {
                    changed = true;
                }
            }
            if(changed) {
                clustering.clear();
                clustering.addAll(newClustering);
                System.arraycopy(curCenter, 0, newCurCenter, 0, curCenter.length);
                indicator = newIndicator.copy();
            } else {
                success = true;
            }
        }
        clusterCenterIndices = new int[numCluster];
        for(i = 0; i < clustering.size(); i++) {
            clusterCenterIndices[i] = clustering.get(i).get(0);
        }
        matrixFai = new Matrix(numBags, numCluster, -1.0);
        int[] intArray = new int[curCenter.length];
        for (i=0; i<intArray.length; ++i)
            intArray[i] = (int) curCenter[i];
        for(i = 0; i < numBags; i++) {
            matrixFai.setMatrix(i, i, 0, matrixFai.getColumnDimension()-1, distanceMatrix.getMatrix(i, i, intArray));
        }
    }
    
    private int minimum(Matrix temp) {
        double min = temp.get(0, 0);
        int index = 0;
        for(int i = 0; i < temp.getColumnDimension(); i++) {
            if(temp.get(0, i) < min) {
                min = temp.get(0, i);
                index = i;
            }
        }
        return index;
    }
    
    private double sumMatrix(Matrix distanceMatrix, ArrayList<Integer> pointsInCluster, int index) {
        double sum = 0.0;
        for(int i = 0; i < pointsInCluster.size(); i++) {
            sum += distanceMatrix.get(index, pointsInCluster.get(i));
        }
        return sum;
    }
    
    private double getDist(Matrix m){
        
        double[][] mat = m.getArray();
        double[][] matTranspose = m.transpose().getArray();
        double[] minVals = new double[numLabels];
        double[] minValsTranspose = new double[numLabels];
        
        //Find minimum value column-wise for the two matrices
        for(int i = 0; i < numLabels; i++){
            minVals[i]= Double.MAX_VALUE;
            minValsTranspose[i] = Double.MAX_VALUE;
            for(int j = 0; j < numLabels;j++){
                minVals[i] = Math.min(minVals[i], mat[i][j]);
                minValsTranspose[i] = Math.min(minValsTranspose[i], matTranspose[i][j]);
            }
        }
        double maxMinVals = Double.MIN_VALUE;
        double maxMinValsTranspose = Double.MIN_VALUE;
        for(int i = 0; i < numLabels; i++){
            maxMinVals = Math.max(maxMinVals, minVals[i]);
            maxMinValsTranspose = Math.max(maxMinValsTranspose, minValsTranspose[i]);
        }
        return Math.max(maxMinVals, maxMinValsTranspose);
    }
    
    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) {
        double[] features = instance.toDoubleArray();
        Matrix innerTestTrain = new Matrix(1, train.getRowDimension(), 0.0);
        Matrix innerTestCenter = new Matrix(1, numLabels, 0.0);
        int numAttributesWithoutLabels = features.length - numLabels;
        Matrix featuresMatrix = new Matrix(features, features.length).getMatrix(0, numAttributesWithoutLabels - 1, 0, 0);
        int i;
        
        Matrix res = null;
        for(i = 0; i < train.getRowDimension(); i++) {
                Matrix b = train.getMatrix(i,i,0,train.getColumnDimension()-1);
                res = featuresMatrix.transpose().times(b.transpose());
                innerTestTrain.setMatrix(0, 0, i, i, res);
        }
        for(i = 0; i < tVec.getRowDimension(); i++) {
            innerTestCenter.setMatrix(0, 0, i, i, 
                            featuresMatrix.transpose().
                                    times(tVec.transpose().
                                            getMatrix(0, tVec.getColumnDimension() -1, i, i)));
        }
        Matrix tempVec = new Matrix(1, clusterCenterIndices.length, 0.0);
        int index;
        Matrix dist;
        double normTestSum = 0.0;
        Matrix normTest = featuresMatrix.arrayTimes(featuresMatrix).transpose();
        for(i = 0; i < normTest.getColumnDimension(); i++) {
            normTestSum += normTest.get(0, i);
        }
        double temp1, temp2, temp3;
        
        for(i = 0; i < clusterCenterIndices.length; i++) {
            index = clusterCenterIndices[i];
            dist = new Matrix(numLabels, numLabels, 0.0);
            for(int m = 0; m < numLabels; m++) {
                for(int n = 0; n < numLabels; n++) {
                    temp1 = normTestSum + normCenter[m] - 2 * innerTestCenter.get(0, m);
                    temp2 = normTrain.get(0, index) + normCenter[n] - 2 * innerTrainCenter.get(index, n);
                    temp3 = -2 * (innerTestTrain.get(0, index) 
                            - innerTestCenter.get(0, n)
                            - innerTrainCenter.get(index, m) 
                            + innerCenterCenter.get(m, n));
                    dist.set(m, n, Math.sqrt(temp1 + temp2 + temp3));
                }
            }
            tempVec.set(0, i, getDist(dist));
        }
        Matrix outputs = tempVec.times(weights);
        boolean[] outputsArr = new boolean[numLabels];
        double[] confidence = new double[numLabels];
        for(i = 0; i < numLabels; i++){
            if(outputs.get(0, i) >= 0) {
                outputsArr[i] = true;
                confidence[i] = outputs.get(0,i);
            } else {
                outputsArr[i] = false;
                confidence[i] = outputs.get(0,i);
            }
        }
        MultiLabelOutput output = new MultiLabelOutput(outputsArr, confidence);
        return output;
    }
    
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INCOLLECTION);
        result.setValue(Field.AUTHOR, "Min-Ling Zhang and Zhi-Hua Zhou");
        result.setValue(Field.TITLE, "Multi-Label Learning by Instance Differentiation");
        result.setValue(Field.PAGES, "669-674");
        result.setValue(Field.BOOKTITLE, "AAAI'07 Proceedings of the 22nd national conference on Artificial intelligence");
        result.setValue(Field.YEAR, "2007");
        result.setValue(Field.VOLUME, "1");
        result.setValue(Field.ISBN, "978-1-57735-323-2");
        return result;
    }
}
