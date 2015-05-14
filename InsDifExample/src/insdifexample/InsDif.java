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
        
    }
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        
        int numInstances = trainingSet.getNumInstances();
        int numLabels = trainingSet.getNumLabels();
        int[] labelIndices = trainingSet.getLabelIndices();
        ArrayList<ArrayList<double[]>> list = new ArrayList<ArrayList<double[]>>();
        // numberofinstances * numbAttributes 
        // numberofinstances * numoflabels
        
        int i, j;
        int numOfAttr = trainingSet.getDataSet().numAttributes();
        double [][] listAttr = new double[numInstances][numOfAttr];
        double [][] listLabels = new double[numInstances][numLabels];
        for(i=0; i< numInstances; i++) {
            for(j=0; j < numOfAttr; j++) {
                listAttr[i][j] = trainingSet.getDataSet().get(i).value(j);
            }
            for(j=numOfAttr; j < numOfAttr; j++ ){
                listLabels[i][j-numOfAttr] = trainingSet.getDataSet().get(i).value(j);
            }
            
        }
        Matrix train = new Matrix(listAttr, numInstances, numOfAttr);
        Matrix trainlabels = new Matrix(listLabels, numInstances, numLabels);
        Matrix normCenter = new Matrix(1,6);
        Matrix normTrain = new Matrix(1,391);
        Matrix tVec = new Matrix (6,72);
        Matrix innerCenterCenter = new Matrix(6,6);
        for(i = 0; i < listLabels[0].length; i++){
            ArrayList<double[]> instancesForClassI = new ArrayList<double[]>();
            for(j = 0; j < numInstances; j++){
                if(listLabels[j][i] == 1){
                    instancesForClassI.add(listAttr[j]);
                }
            }
            list.add(instancesForClassI);
            
        }
        /*
        for(i = 0; i < numLabels; i++) {
            ArrayList<Instance> instances = new ArrayList<Instance>();
            for (j = 0 ; j < trainingSet.getNumInstances(); j++){
                if(trainingSet.getDataSet().get(j).value(labelIndices[i])==1){
                    instances.add(trainingSet.getDataSet().get(j));
                }
            }
            list.add(instances);
            
        }
        */
        double sum[] = new double[numOfAttr];
        double v[][] = new double[numLabels][numOfAttr];
        int k;
        for(i = 0; i < list.size();i++) {
            for(j = 0; j < list.get(i).size(); j++) {
                for(k = 0; k < numOfAttr; k++) {
                    sum[k] += list.get(i).get(j)[k];
                }
            }
            for(k = 0; k < numOfAttr; k++) {
                v[i][k] = sum[k]/list.get(i).size();
            }
        }
        System.out.println("");
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
