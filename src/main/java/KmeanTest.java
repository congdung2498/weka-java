
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author NicoBenic
 */
public class KmeanTest {

    public static void main(String[] args) throws Exception {
        // Load some data
        Instances data = ConverterUtils.DataSource.read("F:\\Data Mining\\BTL\\car-java.arff");
// Create the model
        SimpleKMeans kMeans = new SimpleKMeans();
// We want three clusters
        kMeans.setNumClusters(2);
// Run K-Means
        kMeans.buildClusterer(data);
// Print the centroids
        System.out.println(kMeans.toString());
        Instances centroids = kMeans.getClusterCentroids();
        for (Instance centroid : centroids) {
            System.out.println(centroid);
        }
// Print cluster membership for each instance
        for (Instance point : data) {
            System.out.println(point + " is in cluster " + kMeans.clusterInstance(point));
        }
    }
}
