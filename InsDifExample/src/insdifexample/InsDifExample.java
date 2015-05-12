/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package insdifexample;

import mulan.data.MultiLabelInstances;
import weka.core.Utils;

/**
 *
 * @author geol
 */
public class InsDifExample {
    public static final String TRAIN_FILEPATH = "C:\\Users\\geol\\Dropbox\\advanced ml\\data\\testData\\emotions-train.arff";
    public static final String XML_FILEPATH = "C:\\Users\\geol\\Dropbox\\advanced ml\\data\\testData\\emotions.xml";
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) throws Exception {
        String arffFilename = Utils.getOption("arff", args);
        String xmlFilename = Utils.getOption("xml", args);
        MultiLabelInstances dataset = new MultiLabelInstances(TRAIN_FILEPATH, XML_FILEPATH);
        InsDif ins = new InsDif();
        ins.buildInternal(dataset);
    }
    
}
