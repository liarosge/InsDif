package insdifexample;

import java.net.URL;
import java.util.ArrayList;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
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
        MultiLabelInstances testDataset = new MultiLabelInstances(testFilePath, xmlFilePath);
        InsDif ins = new InsDif(0.1f);
        ins.build(dataset);
        
//        ins.makePrediction(testDataset.getDataSet().get(0));
//        ins.makePredictionInternal(null);
        
        Evaluator ev = new Evaluator();
        ArrayList<Measure> measures = new ArrayList<Measure>();
        measures.add(new AveragePrecision());
        measures.add(new Coverage());
        measures.add(new OneError());
        measures.add(new RankingLoss());
        measures.add(new HammingLoss());
        //Evaluation eval = ev.evaluate(ins, new MultiLabelInstances(testFilePath, xmlFilePath), measures);
        Evaluation eval = ev.evaluate(ins, testDataset, dataset);
        for(Measure measure : eval.getMeasures()){
            System.out.println(measure.getName() + ": " + measure.getValue());
        }
        System.out.println("avpr " + eval.getMeasures().get(4).getValue());
        eval.getClass();
        
    }
    
}
