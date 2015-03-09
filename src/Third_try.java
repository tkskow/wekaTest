import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Loader;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.gui.beans.BatchClassifierEvent;
import weka.gui.beans.PredictionAppender;



public class Third_try {
	
	public static void main(String[] args) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		System.out.print("Enter String");
		String s = br.readLine();
		
		DataSource source = new DataSource(new InputStream(System.in));
		
		Instances test = source.getDataSet();
		
		FilteredClassifier fc = (FilteredClassifier) SerializationHelper.read(new FileInputStream("/home/tk/progging/CS290N/weka-3-6-12/data/testModel.weka.model"));
		
		fc.classifyInstance(test.instance(0));
	}
	
	private Instance inst_co;
	
	public double classify(String string)  {

        // Create attributes to be used with classifiers
        // Test the model
        double result = -1;
        String [] test = string.split(" ");
        try {

            ArrayList<Attribute> attributeList = new ArrayList<Attribute>();
            
            for (int i = 0; i < test.length; i++) {
				
			}
            
            for (String x : test) {
				attributeList.add(new Attribute(x));
			}

            ArrayList<String> classVal = new ArrayList<String>();
            classVal.add("SPAM");
            classVal.add("NEWS");
            classVal.add("SPORTS");


            attributeList.add(new Attribute("@@class@@",classVal));

            Instances data = new Instances("TestInstances",attributeList,0);


            // Create instances for each pollutant with attribute values latitude,
            // longitude and pollutant itself
            inst_co = new DenseInstance(data.numAttributes());
            data.add(inst_co);

            // Set instance's values for the attributes "latitude", "longitude", and
            // "pollutant concentration"
            inst_co.setValue(latitude, lat);
            inst_co.setValue(longitude, lon);
            inst_co.setValue(carbonmonoxide, co);
            // inst_co.setMissing(cluster);

            // load classifier from file
            Classifier cls_co = (Classifier) weka.core.SerializationHelper
                    .read("/CO_J48Model.model");

            result = cls_co.classifyInstance(inst_co);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return result;
    }
	
}
