
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Instance;

/**
 *
 * @author Taha Emara Website: http://www.emaraic.com Email : taha@emaraic.com
 * Created on: Jul 1, 2017 Github link:
 * https://github.com/emara-geek/weka-example
 */
public class Test {

   private Instances trainingData;

    public static void main(String[] args) {
        try {
            Test decisionTree = new Test("F:\\Data Mining\\BTL\\car.arff");
            J48 tree = decisionTree.performTraining();
            System.out.println(tree.toString());
            
            Instance testInstance = decisionTree.
                    getTestInstance("vhigh", "vhigh","2","2","med","low");
            int result = (int) tree.classifyInstance(testInstance);
            String results = decisionTree.trainingData.attribute(6).value(result);
            System.out.println(
                    "Test with: " + testInstance + "  Result: " + results);

//            testInstance = decisionTree.
//                    getTestInstance("Paperback", "no", "historical");
//            result = (int) tree.classifyInstance(testInstance);
//            results = decisionTree.trainingData.attribute(3).value(result);
//            System.out.println(
//                    "Test with: " + testInstance + "  Result: " + results);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public Test(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            trainingData = new Instances(reader);
            trainingData.setClassIndex(trainingData.numAttributes() - 1);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private J48 performTraining() {
        J48 j48 = new J48();
        String[] options = {"-U"};
//        Use unpruned tree. -U
        try {
            j48.setOptions(options);
            j48.buildClassifier(trainingData);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return j48;
    }

    private Instance getTestInstance(
            String buying, String maint , String doors, String persons, String lug_boot, String safety) {
        Instance instance = new DenseInstance(6);
        instance.setDataset(trainingData);
        instance.setValue(trainingData.attribute(0), buying);
        instance.setValue(trainingData.attribute(1), maint);
        instance.setValue(trainingData.attribute(2), doors);
        instance.setValue(trainingData.attribute(3), persons);
        instance.setValue(trainingData.attribute(4), lug_boot);
        instance.setValue(trainingData.attribute(5), safety);
        return instance;
    }
}
