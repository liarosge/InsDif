/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package insdifexperiment;

import java.io.File;
import java.util.ArrayList;
import mulan.classifier.InsDif;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;

/**
 *
 * @author  Agathangelou Sofia
 * @author  Liaros Georgios
 * @author  Paraskevas Eleftherios
 * @author  Tzanakas Alexandros
 * @version 2016.05.21_2218
 */
public class InsDifExperiment {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        File emotions = new File("./data/emotions/");
        File yahoo = new File("./data/yahoo/");
        File yeast = new File("./data/yeast");
        
        File[] listOfFiles;
        ArrayList<File> actualFiles = new ArrayList<>();
        
        // Add the datasets
        actualFiles.add(emotions);
        actualFiles.add(yahoo);
        actualFiles.add(yeast);
        
        String name;
        String experiment;
        Evaluation eval;
        MultiLabelInstances trainDataset;
        MultiLabelInstances testDataset;        
        InsDif ins;
        Evaluator ev;
        
        for (File actualFile : actualFiles) {
            listOfFiles = actualFile.listFiles();
            experiment = actualFile.getName();
            System.out.println(experiment + " experiment");
            System.out.println("********************\n");
            for (File listOfFile : listOfFiles) {
                if (listOfFile.getName().contains(".xml")) {
                    
                    name = listOfFile.getName().substring(0, listOfFile.getName().indexOf("."));
                    
                    trainDataset = new MultiLabelInstances("./data/" 
                            + experiment + "/" + name + "-train.arff", "./data/" 
                            + experiment + "/" + name + ".xml");
                    testDataset = new MultiLabelInstances("./data/" 
                            + experiment + "/" + name + "-test.arff", "./data/" 
                            + experiment + "/" + name + ".xml");
                    ins = new InsDif(0.1f);
                    ins.build(trainDataset);
                    ev = new Evaluator();
                    eval = ev.evaluate(ins, testDataset, trainDataset);
                    eval.getMeasures().stream().forEach((measure) -> {
                        System.out.println(measure.getName() + ": " + measure.getValue());
                    });
                    System.out.println("avpr " + eval.getMeasures().get(4).getValue());
                }
            }
            System.out.println("");
        }
    }
    
}
