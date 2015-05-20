package mulan.classifier;

import junit.framework.Assert;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author  Agathangelou Sofia
 * @author  Liaros Georgios
 * @author  Paraskevas Eleftherios
 * @author  Tzanakas Alexandros
 * @version 2015.05.20
 * 
 */
public class InsDifITest {
    
    private final String trainPath = "./data/testData/emotions-train.arff";
    private final String testPath = "./data/testData/emotions-test.arff";
    private final String xmlPath = "./data/testData/emotions.xml";
    
    public InsDifITest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
        
        
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of buildInternal method, of class InsDif.
     * @throws java.lang.Exception
     */
    @Test
    public void testBuildInternal() throws Exception {
        MultiLabelInstances trainingSet = new MultiLabelInstances(trainPath, xmlPath);
        InsDif learner = new InsDif();

        Assert.assertNull(learner.labelIndices);
        Assert.assertNull(learner.featureIndices);
        Assert.assertEquals(0, learner.numLabels);

        learner.build(trainingSet);

        Assert.assertNotNull(learner.labelIndices);
        Assert.assertNotNull(learner.featureIndices);
        Assert.assertEquals(trainingSet.getNumLabels(), learner.numLabels);

    }

    /**
     * Test of makePredictionInternal method, of class InsDif.
     * @throws mulan.data.InvalidDataFormatException
     */
    @Test
    public void testMakePredictionInternal() throws InvalidDataFormatException, Exception {
        MultiLabelInstances trainingSet = new MultiLabelInstances(trainPath, xmlPath);
        MultiLabelInstances testingSet = new MultiLabelInstances(testPath, xmlPath);
        InsDif instance = new InsDif();
        instance.build(trainingSet);
        InsDif instance2 = (InsDif) instance.makeCopy();
        for(int i=0; i < testingSet.getDataSet().numInstances(); i++){
            MultiLabelOutput mlo1 = instance.makePrediction(testingSet.getDataSet().instance(i));
            MultiLabelOutput mlo2= instance2.makePrediction(testingSet.getDataSet().instance(i));
            assertEquals(mlo1,mlo2);
        }
    }

    /**
     * Test of getTechnicalInformation method, of class InsDif.
     */
    @Test
    public void testGetTechnicalInformation() {
        InsDif learner = new InsDif();
       Assert.assertNotNull(learner.getTechnicalInformation());
    }
    
}
