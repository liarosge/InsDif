/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package insdifexample;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import java.util.ArrayList;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.matrix.Matrix;
/**
 *
 * @author 
 */
public class InsDif extends MultiLabelLearnerBase implements MultiLabelLearner {
    
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
        
        /*
        *   *************************************
        *   Step 1 - Create the prototype vectors
        *   *************************************
        */
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
                System.out.println("list.get(i).size = " + list.get(i).size());
                prototypeVectors[i][k] = sum[k]/list.get(i).size();
            }
        }
        //Matlab check: prototypevectors correct!
        System.out.println("");
        int numClusters = (int) (numOfInstances*ratio);
        Matrix train = new Matrix(listAttr);
        Matrix trainlabels = new Matrix(listLabels, numOfInstances, numLabels);
        double[] normCenter = new double[numLabels];
        Matrix normTrain = new Matrix(1,numOfInstances); //1x391
        Matrix tVec = new Matrix(prototypeVectors); //6x72
//        tVec.print(10,3);
        Matrix innerCenterCenter = new Matrix(numLabels,numLabels);// 6x6
        Matrix innerTrainCenter = new Matrix (numOfInstances, numLabels); //391x6
        Matrix innerTrainTrain = new Matrix (numOfInstances, numOfInstances); //391x391
        Matrix matrixFai = new Matrix(numOfInstances, numClusters); //391x39 for ratio =0.1
        Matrix distanceMatrix = new Matrix(numOfInstances, numOfInstances);
        //calc norm_train
        Matrix s = train.arrayTimes(train).transpose();
        System.out.println(s.getRowDimension());
        System.out.println(s.getColumnDimension());
        for(i = 0; i < s.getRowDimension(); i++){
            Matrix temp = s.getMatrix(i,i,0,s.getColumnDimension()-1);
            normTrain.plusEquals(temp);
        }
        //Matlab check: normTrain correct!
        
        //calc norm center
        for(i = 0; i < numLabels; i++){
            Matrix a = tVec.getMatrix(i,i,0,tVec.getColumnDimension()-1);
            Matrix b = tVec.getMatrix(i,i,0,tVec.getColumnDimension()-1).transpose();
            normCenter[i] = a.times(b).get(0,0);
        }
        //Matlab check: normCenter correct!
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
//        innerTrainTrain.print(10,3);
        //Matlab check: innerTrainTrain correct!
        //calc innerTrainCenter
        for(i = 0; i < numOfInstances; i++){
            for(j = 0; j < numLabels; j++){
                Matrix a = train.getMatrix(i,i,0,train.getColumnDimension()-1);
                Matrix b = tVec.getMatrix(j,j,0, tVec.getColumnDimension()-1).transpose();
                innerTrainCenter.set(i,j,a.times(b).get(0,0));
            }
        }
//        innerTrainCenter.print(10,3);
        //Matlab check: innerTrainCenter correct!
        for(i = 0; i < numLabels; i++){
            for(j = 0 ; j < numLabels; j++){
                Matrix a = tVec.getMatrix(i, i, 0, tVec.getColumnDimension()-1);
                Matrix b = tVec.getMatrix(j, j, 0, tVec.getColumnDimension()-1).transpose();
                innerCenterCenter.set(i, j, a.times(b).get(0,0));
            }
        }
//        innerCenterCenter.print(10,3);
        //Matlab check: innerCenteCenter correct!
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
        //Matlab check: seems good so far
        //TODO: MIML_cluster
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
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
