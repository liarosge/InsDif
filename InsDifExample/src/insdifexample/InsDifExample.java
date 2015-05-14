/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package insdifexample;

import java.net.URL;
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
        trainFilePath = InsDifExample.class.getResource("../resources/emotions-train.arff").getFile();
        testFilePath = InsDifExample.class.getResource("../resources/emotions-test.arff").getFile();
        xmlFilePath = InsDifExample.class.getResource("../resources/emotions.xml").getFile();
        MultiLabelInstances dataset = new MultiLabelInstances(trainFilePath, xmlFilePath);
        InsDif ins = new InsDif(0.1f);
        ins.buildInternal(dataset);
    }
    
}
