
import java.io.FileInputStream;
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
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class First_try {

	public static void main(String[] args) throws Exception {
		//treningsfilen
		DataSource source = new DataSource("/home/tk/progging/CS290N/weka-3-6-12/data/5000_tweets2.arff");
		Instances data = source.getDataSet();
		//testfilen
		DataSource source2 = new DataSource("/home/tk/progging/CS290N/weka-3-6-12/data/testset2.arff");
		Instances test = source2.getDataSet();
		
		//noe som var i eksempelet
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		
		if (test.classIndex() == -1)
			test.setClassIndex(test.numAttributes() - 1);
		
		// var i eksempelet
		 Remove rm = new Remove();
		 rm.setAttributeIndices("1");  // remove 1st attribute
		 
		 //stemmer
		 IteratedLovinsStemmer ils = new IteratedLovinsStemmer();
		 
		 //wordtokenizer for StringToWordVector
		 WordTokenizer wt = new WordTokenizer();
		 wt.setDelimiters(".,;:'\"()?!");
		 
		 //oppretter en StringToWordVector for å bruke i fieldclassifier
		 StringToWordVector stwv = new StringToWordVector();
		 stwv.setTFTransform(true);
		 stwv.setIDFTransform(true);
		 stwv.setAttributeIndices("first-last");
		 //stwv.setAttributeNamePrefix(null);		//bruker default, vet ikke hva som skal fylles inn
		 stwv.setDoNotOperateOnPerClassBasis(false);
		 stwv.setInvertSelection(false);
		 stwv.setLowerCaseTokens(false);
		 stwv.setMinTermFreq(1);
		 //stwv.setNormalizeDocLength(null);		//bruker default, vet ikke hva som skal fylles inn
		 stwv.setOutputWordCounts(true);
		 stwv.setPeriodicPruning(-1.0);
		 stwv.setStemmer(ils);
		 //stwv.setStopwords(null);		//bruker default, vet ikke hvordan jeg skal fylle inn
		 stwv.setTokenizer(wt);
		 stwv.setUseStoplist(true);
		 stwv.setWordsToKeep(1000);
		 
		 //oppretter ett J48 tre med verdiene samme som i gui
		 J48 j48 = new J48();
		 j48.setBinarySplits(false);
		 j48.setConfidenceFactor((float) 0.25);
		 j48.setDebug(false);
		 j48.setMinNumObj(2);
		 j48.setNumFolds(3);
		 j48.setReducedErrorPruning(false);
		 j48.setSaveInstanceData(false);
		 j48.setSeed(1);
		 j48.setSubtreeRaising(true);
		 j48.setUnpruned(false);
		 j48.setUseLaplace(false);
		 
		 // FilterClassifier med verdier.
		 FilteredClassifier fc = new FilteredClassifier();
		 fc.setFilter(stwv);
		 fc.setClassifier(j48);
		 fc.setDebug(false);
		 // train and make predictions
		 fc.buildClassifier(data);
		 
		 //test
		 Evaluation eval = new Evaluation(data);
		 eval.evaluateModel(fc, test);
		 System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		 
	}
}
