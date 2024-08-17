/**
 * This class represents a set of training, validation or testing instances.
 *
 * These instances are loaded from a text file. Each row
 * of the txt file is stored as an Sequence object, which has it's values and
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
import java.util.Collections;
import java.util.List;
import java.util.HashSet;

import util.Log;

public class SequenceDataSet implements DataSet {
    //A nice name to describe this data set (which will help with nice output messages)
    String name;

    //The filenames used in this data set
    String filename;

    /**
     * This is the maximum length of any sequence in the data
     * set. We can use this to bound how long our arrays need
     * to be in our nodes and edges.
     */
    int maxSequenceLength;


    //This will hold the instances (expected outputs and inputs) for the txt input file
    //Each line of the text file will be a separate instance
    List<CharacterSequence> instances;

    /**
     * This creates a new SequenceDataSet object provided a filename
     *
     * @param name is a nice name to describe the data set
     * @param filenames is list of filenames to load the sequences from
     */
    public SequenceDataSet(String name, String filename) {
        this.name = name;
        this.filename = filename;

        //the instances List interface will be instantiated as an ArrayList
        instances = new ArrayList<CharacterSequence>();

        try {
            Log.info("creating a DataSet called '" + name + "' from the following txtFile: '" + filename + "'");

            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(filename)));

            String readLine = "";

            int line = 0;
            Log.trace("'" + filename + "' contents are:");
            //read the file line by line
            while ((readLine = bufferedReader.readLine()) != null) {
                //increment the line so we know what line we're reading from
                line++;

                Log.trace(readLine); //print out each line of the file if the -DLOG_LEVEL=DEBUG system property is set on the command line

                //generate a new CharacterSequence from the line read from the file
                try {
                    CharacterSequence sequence = new CharacterSequence(readLine);
                    instances.add(sequence);

                    int sequenceLength = sequence.getLength();
                    Log.trace("created a sequence with length: " + sequenceLength);
                    if (sequenceLength > maxSequenceLength) maxSequenceLength = sequenceLength;
                } catch (SequenceException e) {
                    Log.fatal("Could not convert sequence: " + readLine);
                    Log.fatal("threw exception: " + e);
                    e.printStackTrace();
                    System.exit(1);
                }
            }
            Log.info("max sequence length of the data set was: " + maxSequenceLength);

        } catch (IOException e) {
            Log.fatal("ERROR opening DataSet file: '" + filename + "'");
            e.printStackTrace();
            System.exit(1);
        }
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
     * Gets the length of the longest sequence in the instances
     *
     * @return the length of the longest sequence in the instances
     */
    public int getMaxLength() {
        return maxSequenceLength;
    }


    /**
     * Gets the number of instances in this DataSet
     *
     * @return the number of instances in this DataSet
     */
    public int getNumberSequences() {
        return instances.size();
    }

    /**
     * Gets the number of values in the output data in the instances
     * of this DataSet. The different possible characters are (we will count <unk> 
     * as a single character):
     *
     * ' '
     * '#'
     * '$'
     * '&'
     * '''
     * '*'
     * '-'
     * '.'
     * '/'
     * '\'
     * N        -- most numbers have been replaced with N
     * <unk>    -- rare words like names have been replaced with <unk>, we will use ? as a standin for <unk>
     * 0-9      -- numbers
     * a-z      -- all characters have been lowercased
     * 
     * for a total of 48 characters
     *
     * @return the number of possible outputs per instance
     */
    public int getNumberOutputs() {
       return 48;
    }

    /**
     * Gets the number of values in the input data in the instances
     * of this DataSet
     *
     * @return the number of inputs per instance, which is the number of total characters 
     */
    public int getNumberInputs() {
        return 48;
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
    public Sequence getSequence(int position) {
        Log.trace("Getting instance[" + position + "] from '" + name + "'");
        return instances.get(position);
    }

    /**
     * This gets a consecutive set of instances from the instances
     * ArrayList. position should be >= 0 and numberOfSequences should
     * be >= 1.
     *
     * @param position the position of the first instance to return
     * @param numberOfSequences is how many instances to return. If 
     * position + numberOfSequences is > than instances.size() it will
     * return the remaining instances in the instances ArrayList.
     *
     * @return An ArrayList of the instances specified by position and
     * numberOfSequences. Its size will be <= numberOfSequences.
     */
    public List<Sequence> getSequences(int position, int numberOfSequences) {
        int endIndex = position + numberOfSequences;
        if (endIndex > instances.size()) endIndex = instances.size();

        Log.trace("Getting instances[" + position + " to " + endIndex + "] from '" + name + "'");
        List<CharacterSequence> subList = instances.subList(position, endIndex);

        List<Sequence> castedList = new ArrayList<Sequence>();
        for (int i = 0; i < subList.size(); i++) {
            castedList.add(subList.get(i));
        }
        return castedList;
    }

    /**
     * This gets the entire set of instances from the instances
     * ArrayList. 
     *
     * @return The ArrayList of the instances of this data set.
     */
    public List<Sequence> getSequences() {
        List<Sequence> castedList = new ArrayList<Sequence>();
        for (int i = 0; i < instances.size(); i++) {
            castedList.add(instances.get(i));
        }
        return castedList;
    }

    /**
     * This gets the entire set of instances from the instances
     * ArrayList. 
     *
     * @return The ArrayList of the instances of this data set.
     */
    public List<CharacterSequence> getCharacterSequences() {
        return instances;
    }


}
