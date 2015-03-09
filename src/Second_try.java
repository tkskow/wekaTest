
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.ObjectInputStream;

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
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Second_try {
	
	public static void main(String[] args) throws FileNotFoundException, Exception {
		FilteredClassifier fc = (FilteredClassifier) SerializationHelper.read(new FileInputStream("/home/tk/progging/CS290N/weka-3-6-12/data/testModel.weka.model"));
		
		DataSource source2 = new DataSource("/home/tk/progging/CS290N/weka-3-6-12/data/5000_tweets2.arff");
		Instances test = source2.getDataSet();
		
		DataSource source3 = new DataSource("/home/tk/progging/CS290N/test.arff");
		Instances test2 = source3.getDataSet();
		
		DataSource source4 = new DataSource("/home/tk/progging/CS290N/weka-3-6-12/data/testset2.arff");
		Instances test3 = source4.getDataSet();
		
		if (test.classIndex() == -1)
			test.setClassIndex(test.numAttributes() - 1);
		
		if (test2.classIndex() == -1)
			test2.setClassIndex(test2.numAttributes() - 1);
		
		if (test3.classIndex() == -1)
			test3.setClassIndex(test3.numAttributes() - 1);

		
		
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(fc, test);
		 
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		System.out.println(eval.toMatrixString());
		
		int correct = 0;
		int wrong = 0;
		int antallSpam = 0;
		int resten = 0;
		for (int i = 0; i < test3.numInstances(); i++) {
			if (test3.classAttribute().value((int) test3.instance(i).classValue()) == test3.classAttribute().value((int) 0.0)) {
				antallSpam++;
			}
			else resten++;
		}
		for (int i = 0; i < test3.numInstances(); i++) {
			double pred = fc.classifyInstance(test3.instance(i));
//			System.out.print(", actual: " + test3.classAttribute().value((int) test3.instance(i).classValue()));
//			System.out.println(", predicted: " + test3.classAttribute().value((int) pred));
			if (test3.classAttribute().value((int) test3.instance(i).classValue()) == test3.classAttribute().value((int) pred)) {
				correct++;
			}
			else wrong++;
		}
		for (int i = 0; i < test2.numInstances(); i++) {
			double pred = fc.classifyInstance(test2.instance(i));
			System.out.print(", actual: " + test2.classAttribute().value((int) test2.instance(i).classValue()));
			System.out.println(", predicted: " + test2.classAttribute().value((int) pred));
//			if (test2.classAttribute().value((int) test2.instance(i).classValue()) == test2.classAttribute().value((int) pred)) {
//				correct++;
//			}
//			else wrong++;
		}
		
		
		System.out.println("Correct %: " + ((double) correct/test3.numInstances())*100);
		System.out.println("Wrong %: " + ((double) wrong/test3.numInstances())*100);
		
		System.out.println(antallSpam);
		System.out.println(resten);

		 
		

	}
}
