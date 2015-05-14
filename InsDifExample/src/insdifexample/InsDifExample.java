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
    public static String trainFilePath;// = "C:\\Users\\geol\\Dropbox\\advanced ml\\data\\testData\\emotions-train.arff";
    public static String testFilePath;
    public static String xmlFilePath;// = "C:\\Users\\geol\\Dropbox\\advanced ml\\data\\testData\\emotions.xml";
    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) throws Exception {
        ClassLoader classLoader = new InsDifExample().getClass().getClassLoader();
        trainFilePath = classLoader.getResource("resources/emotions-train.arrf").getFile();
        testFilePath = classLoader.getResource("resources/emotions-test.arff").getFile();
        xmlFilePath = classLoader.getResource("resources/emotions.xml").getFile();
        MultiLabelInstances dataset = new MultiLabelInstances(trainFilePath, xmlFilePath);
        InsDif ins = new InsDif();
        ins.buildInternal(dataset);
    }
    
}
