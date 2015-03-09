/**
 * Java program for classifying short text messages into two classes.
 */

import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.*;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;
import java.util.*;

public class MessageClassifier implements Serializable {

  /* Our (rather arbitrary) set of keywords. */
  private final String[] m_Keywords = {"product", "only", "offer", "great", "amazing", "phantastic", "opportunity", "buy", "now"};

  /* The training data. */
  private Instances m_Data = null;

  /* The filter. */
  private Filter m_Filter = new StringToWordVector(1000);

  /* The classifier. */
  private Classifier m_Classifier = new FilteredClassifier();

  /**
   * Constructs empty training dataset.
   */
  public MessageClassifier() throws Exception {

    String nameOfDataset = "MessageClassificationProblem";

    // Create numeric attributes.
    FastVector attributes = new FastVector(m_Keywords.length + 1);
    for (int i = 0 ; i < m_Keywords.length; i++) {
      attributes.addElement(new Attribute(m_Keywords[i]));
    }

    // Add class attribute.
    FastVector classValues = new FastVector(2);
    classValues.addElement("miss");
    classValues.addElement("hit");
    attributes.addElement(new Attribute("Class", classValues));

    // Create dataset with initial capacity of 100, and set index of class.
    m_Data = new Instances(nameOfDataset, attributes, 100);
    m_Data.setClassIndex(m_Data.numAttributes() - 1);
  }

  /**
   * Updates model using the given training message.
   */
  public void updateModel(String message, String classValue) 
    throws Exception {

    // Convert message string into instance.
    Instance instance = makeInstance(cleanupString(message));

    // Add class value to instance.
    instance.setClassValue(classValue);

    // Add instance to training data.
    m_Data.add(instance);

    // Use filter.
    m_Filter.input(instance);
    Instances filteredData = Filter.useFilter(m_Data, m_Filter);

    // Rebuild classifier.
    m_Classifier.buildClassifier(filteredData);
  }

  /**
   * Classifies a given message.
   */
  public void classifyMessage(String message) throws Exception {

    // Check if classifier has been built.
    if (m_Data.numInstances() == 0) {
      throw new Exception("No classifier available.");
    }

    // Convert message string into instance.
    Instance instance = makeInstance(cleanupString(message));

    // Filter instance.
    m_Filter.input(instance);
    Instance filteredInstance = m_Filter.output();

    // Get index of predicted class value.
    double predicted = m_Classifier.classifyInstance(filteredInstance);

    // Classify instance.
    System.err.println("Message classified as : " +
                    m_Data.classAttribute().value((int)predicted));
  }

  /**
   * Method that converts a text message into an instance.
   */
  private Instance makeInstance(String messageText) {

    StringTokenizer tokenizer = new StringTokenizer(messageText);
    Instance instance = new Instance(m_Keywords.length + 1);
    String token;

    // Initialize counts to zero.
    for (int i = 0; i < m_Keywords.length; i++) {
      instance.setValue(i, 0);
    }

    // Compute attribute values.
    while (tokenizer.hasMoreTokens()) {
      token = tokenizer.nextToken();
      for (int i = 0; i < m_Keywords.length; i++) {
        if (token.equals(m_Keywords[i])) {
          instance.setValue(i, instance.value(i) + 1.0);
          break;
        }
      }
    }

    // Give instance access to attribute information from the dataset.
    instance.setDataset(m_Data);

    return instance;
  }

  /**
   * Method that deletes all non-letters from a string, and lowercases it.
   */
  private String cleanupString(String messageText) {

    char[] result = new char[messageText.length()];
    int position = 0;

    for (int i = 0; i < messageText.length(); i++) {
      if (Character.isLetter(messageText.charAt(i)) ||
        Character.isWhitespace(messageText.charAt(i))) {
        result[position++] = Character.toLowerCase(messageText.charAt(i));
      }
    }
    return new String(result);
  }

  /**
   * Main method.
   */
  public static void main(String[] args) {
	String [] options = "/home/tk/progging/CS290N/test.txt" 
	  
    MessageClassifier messageCl;
    byte[] charArray;

    try {

      // Read message file into string.
      String messageFileString = Utils.getOption('m', options);
      if (messageFileString.length() != 0) {
        FileInputStream messageFile = new FileInputStream(messageFileString);
        int numChars = messageFile.available();
        charArray = new byte[numChars];
        messageFile.read(charArray);
        messageFile.close();
      } else {
        throw new Exception ("Name of message file not provided.");
      }

      // Check if class value is given.
      String classValue = Utils.getOption('c', options);

      // Check for model file. If existent, read it, otherwise create new
      // one.
      String modelFileString = Utils.getOption('t', options);
      if (modelFileString.length() != 0) {
        try {
          FileInputStream modelInFile = new FileInputStream(modelFileString);
          ObjectInputStream modelInObjectFile = 
            new ObjectInputStream(modelInFile);
          messageCl = (MessageClassifier) modelInObjectFile.readObject();
          modelInFile.close();
        } catch (FileNotFoundException e) {
          messageCl = new MessageClassifier();
        }
      } else {
        throw new Exception ("Name of data file not provided.");
      }

      // Check if there are any options left
      Utils.checkForRemainingOptions(options);

      // Process message.
      if (classValue.length() != 0) {
        messageCl.updateModel(new String(charArray), classValue);
      } else {
        messageCl.classifyMessage(new String(charArray));
      }

      // If class has been given, updated message classifier must be saved
      if (classValue.length() != 0) {
        FileOutputStream modelOutFile =
          new FileOutputStream(modelFileString);
        ObjectOutputStream modelOutObjectFile = 
          new ObjectOutputStream(modelOutFile);
        modelOutObjectFile.writeObject(messageCl);
        modelOutObjectFile.flush();
        modelOutFile.close();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}