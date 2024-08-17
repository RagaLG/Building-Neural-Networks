/**
 * This class represents a set of training, validation or testing instances.
 *
 * These instances are loaded from a text file. Each row
 * of the txt file is stored as an Instance object, which has it's values and
 * expected class/output.
 * Each row in the file will have have ':' which will separate the expected
 * outputs from the expected inputs, the values of the outputs and inputs
 * will be separated by commas, e.g.:
 *
 * output1,output2,output3:input1,input2,input3,input4.input5
 *
 * Empty lines and lines beginning with the '#' character  are ignored.
 */

package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.HashSet;

import util.Log;

public class DataSet {
    //A nice name to describe this data set (which will help with nice output messages)
    String name;

    //The filename for this dataset
    String filename;

    //This will hold the instances (expected outputs and inputs) for the txt input file
    List<Instance> instances;

    //this will hold how many output values there are per instance, this needs
    //to be the same for all instances
    int numberOutputs;

    //this will hold how many input values there are per instance, this needs
    //to be the same for all instances
    int numberInputs;


    //this will hold how many classes here are in the data, for example in iris
    //there are 3 (because there are 3 potential class values) and in mushroom
    //there are 2 (because there are 2 potential class values)
    int numberClasses;


    /**
     * This creates a new DataSet object provided a filename
     *
     * @param name is a nice name to describe the data set
     * @param filename is the name of the TXT file to load the instances from
     */
    public DataSet(String name, String filename) {
        this.name = name;
        this.filename = filename;

        numberOutputs = -1;
        numberInputs = -1;

        //the instances List interface will be instantiated as an ArrayList
        instances = new ArrayList<Instance>();

        //We will use potentialOutputs as a Set which means it will only contain 
        //unique values, so the number of classes are the total number of unique outputs
        HashSet<Double> potentialOutputs = new HashSet<Double>();

        try {
            Log.info("creating a DataSet called '" + name + "' from the following txtFile: '" + filename + "'");

            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(filename)));

            String readLine = "";

            int line = 0;
            Log.debug("'" + filename + "' contents are:");
            //read the file line by line
            while ((readLine = bufferedReader.readLine()) != null) {
                //increment the line so we know what line we're reading from
                line++;

                Log.debug(readLine); //print out each line of the file if the -DLOG_LEVEL=DEBUG system property is set on the command line

                if (readLine.length() == 0 || readLine.charAt(0) == '#') {
                    //empty lines are skipped, as well as lines beginning 
                    //with the '#' character, these are comments and also skipped
                    continue;
                }

                //split the line into the output and input parts
                String[] outputInput = readLine.split(":");

                if (outputInput.length != 2) {
                    throw new IOException("Line " + line + " of '" + filename + "' was not properly formatted, it should contain exactly one ':' character dividing the expected outputs and inputs.");
                }

                //split up the output and input parts into the individual
                //values
                String[] outputStrings = outputInput[0].split(",");
                if (outputStrings.length == 0) {
                    throw new IOException("Line " + line + " of '" + filename + "' was not properly formatted, it did not contain any expected outputs.");
                }

                //validate that the number of outputs is correct
                if (numberOutputs == -1) {
                    //we havent determined the number of outputs yet, this will be determine on the
                    //first instance line we read
                    numberOutputs = outputStrings.length;
                } else if (outputStrings.length != numberOutputs) {
                    //this line has a different number of outputs than all previous instances so throw
                    //an exception
                    throw new IOException("Line " + line + " of '" + filename + "' was not properly formatted, it had a different number of outputs (" + outputStrings.length + ") than the number of outputs in previously read instances (" + numberOutputs + ")");
                }

                String[] inputStrings = outputInput[1].split(",");
                if (inputStrings.length == 0) {
                    throw new IOException("Line " + line + " of '" + filename + "' was not properly formatted, it did not contain any expected inputs.");
                }

                //validate that the number of inputs is correct
                if (numberInputs == -1) {
                    //we havent determined the number of inputs yet, this will be determine on the
                    //first instance line we read
                    numberInputs = inputStrings.length;
                } else if (inputStrings.length != numberInputs) {
                    //this line has a different number of inputs than all previous instances so throw
                    //an exception
                    throw new IOException("Line " + line + " of '" + filename + "' was not properly formatted, it had a different number of inputs (" + inputStrings.length + ") than the number of inputs in previously read instances (" + numberInputs + ")");
                }

                //convert the output strings to doubles
                double[] outputs = new double[outputStrings.length];

                for (int i = 0; i < outputs.length; i++) {
                    //Double.parseDouble converts a string representation of
                    //a double to a double
                    outputs[i] = Double.parseDouble(outputStrings[i]);
                    potentialOutputs.add(outputs[i]);
                }

                //convert the input strings to doubles
                double[] inputs = new double[inputStrings.length];
                for (int i = 0; i < inputs.length; i++) {
                    inputs[i] = Double.parseDouble(inputStrings[i]);
                }

                //now that we have parsed the expected output and inputs we
                //can create a new Instance object and add it to the instances
                //ArrayList
                instances.add(new Instance(outputs, inputs));
            }

            //potentialOutputs is a Set so it will only contain unique values,
            //so the number of classes are the total number of unique outputs
            numberClasses = potentialOutputs.size();
            Log.info("Dataset had " + numberOutputs + " outputs with " + numberClasses + " classes, " + numberInputs + " inputs and " + instances.size() + " instances.");

        } catch (IOException e) {
            Log.fatal("ERROR opening DataSet file: '" + filename + "'");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public double[] getInputMeans() {
        //TODO: You need to implement this for PA1-4
        //You need to calculate the mean of each input column
        //Create an array of doubles to hold the means of each input
        double[] meansInput = new double[numberInputs];
        for (int i = 0; i < this.numberInputs; i++) {
            meansInput[i] = 0.0;
        }
        //Iterate through each instance and sum the inputs
        for (Instance instance : instances) {
            for (int i = 0; i < this.numberInputs; i++) {
                meansInput[i] += instance.inputs[i];
            }
        }
        //Divide the sum of the inputs by the number of instances to get the mean
        for (int i = 0; i < this.numberInputs; i++) {
            meansInput[i] = meansInput[i]/instances.size();
        }
        return meansInput;
    }

    public double[] getInputStandardDeviations() {
        //TODO: You need to implement this for PA1-4
        //You need to calculate the standardDeviation of each input column,
        //hint: you can re-use your getInputMeans function for this
        
        double[] meansInput = getInputMeans(); //get the means of the inputs
        double[] squares = new double[numberInputs]; //create an array for standard deviations
        double[] stdDeviations = new double[numberInputs];
        // Initialize the array to 0.0
        Arrays.fill(stdDeviations, 0.0);
        //Iterate through each instance and sum the squared differences from the mean
        for(Instance instance : instances){
            for(int i=0; i < numberInputs; i++){
                squares[i] += Math.pow(instance.inputs[i] - meansInput[i], 2);
            }
        }
        //Divide the sum of the squared differences by the number of instances and take the square root
        for(int i=0; i < this.numberInputs; i++){
            stdDeviations[i] = Math.sqrt(squares[i]/(instances.size()-1));
        }
        return stdDeviations;
    }


    public void normalize(double[] inputMeans, double[] inputStandardDeviations) {
        //TODO: You need to implement this for PA1-4
        //You should subtract the mean of each column and divide by
        //the standard deviation
        List<Instance> normalizedInstances = new ArrayList<Instance>();
        for(Instance instance : instances){
            double[] normalizedInputs = instance.inputs;
            for(int i=0; i < normalizedInputs.length; i++){
                normalizedInputs[i] = (normalizedInputs[i] - inputMeans[i])/inputStandardDeviations[i];
            }
            normalizedInstances.add(new Instance(instance.expectedOutputs, normalizedInputs));
            //System.out.println(Arrays.toString(instance.inputs));
        }
        this.instances = normalizedInstances;
    }

    /**
     * Gets the nice human readable name of this DataSet
     * 
     * @return the name of this dataset
     */
    public String getName() {
        return name;
    }

    /**
     * Gets the number of instances in this DataSet
     *
     * @return the number of instances in this DataSet
     */
    public int getNumberInstances() {
        return instances.size();
    }

    /**
     * Gets the number of values in the input data in the instances
     * of this DataSet
     *
     * @return the number of inputs per instance
     */
    public int getNumberInputs() {
        return numberInputs;
    }

    /**
     * Gets the number of values in the output data in the instances
     * of this DataSet
     *
     * @return the number of outputs per instance
     */
    public int getNumberOutputs() {
        return numberOutputs;
    }

    /**
     * Gets the number of values in the output data in the instances
     * of this DataSet
     *
     * @return the number of outputs per instance
     */
    public int getNumberClasses() {
        return numberClasses;
    }


    /**
     * This randomly shuffles the orders of the instances in the 
     * instances ArrayList. This will be useful when we are implementing 
     * different versions of stochastic backpropagation.
     */
    public void shuffle() {
        Log.trace("Shuffling '" + name + "'");
        Collections.shuffle(instances);
    }

    /**
     * This gets the instance in the specified position in the
     * instances ArrayList.
     *
     * @param position is the position of the instance to retrieve
     * from the instances ArrayList. position should be >= 0 and <
     * instances.size() or else it will throw an ArrayOutOfBounds 
     * exception.
     */
    public Instance getInstance(int position) {
        Log.trace("Getting instance[" + position + "] from '" + name + "'");
        return instances.get(position);
    }

    /**
     * This gets a consecutive set of instances from the instances
     * ArrayList. position should be >= 0 and numberOfInstances should
     * be >= 1.
     *
     * @param position the position of the first instance to return
     * @param numberOfInstances is how many instances to return. If 
     * position + numberOfInstances is > than instances.size() it will
     * return the remaining instances in the instances ArrayList.
     *
     * @return An ArrayList of the instances specified by position and
     * numberOfInstances. Its size will be <= numberOfInstances.
     */
    public List<Instance> getInstances(int position, int numberOfInstances) {
        int endIndex = position + numberOfInstances;
        if (endIndex > instances.size()) endIndex = instances.size();

        Log.trace("Getting instances[" + position + " to " + endIndex + "] from '" + name + "'");
        return instances.subList(position, endIndex);
    }

    /**
     * This gets the entire set of instances from the instances
     * ArrayList. 
     *
     * @return The ArrayList of the instances of this data set.
     */
    public List<Instance> getInstances() {
        return instances;
    }
}
