/**
 * This class represents a set of training, validation or testing timeSeries.
 *
 * These timeSeries are loaded from a text file. Each row
 * of the txt file is stored as an TimeSeries object, which has it's values and
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

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashSet;

import util.Log;

public class TimeSeriesDataSet implements DataSet {
    //A nice name to describe this data set (which will help with nice output messages)
    String name;

    //The filenames used in this data set
    String[] filenames;

    //the column names for the input parameters
    String[] inputNames;

    //the column names for the output parameters
    String[] outputNames;

    /**
     * This is the maximum length of any timeSeries in the data
     * set. We can use this to bound how long our arrays need
     * to be in our nodes and edges.
     */
    int maxTimeSeriesLength;


    //This will hold each timeSeries (expected outputs and inputs) for the txt input file
    //Each line of the text file will be a separate timeSeries
    List<TimeSeries> timeSeries;


    /**
     * This creates a new TimeSeriesDataSet object provided a filename
     *
     * @param name is a nice name to describe the data set
     * @param filenames is list of filenames to load the timeSeries from
     */
    public TimeSeriesDataSet(String name, String[] filenames, String[] inputNames, String[] outputNames) {
        this.name = name;
        this.filenames = filenames;
        this.inputNames = inputNames;
        this.outputNames = outputNames;

        //the timeSeries List interface will be instantiated as an ArrayList
        timeSeries = new ArrayList<TimeSeries>();

        Log.info("creating a DataSet called '" + name + "' from the following csv files: '" + Arrays.toString(filenames) + "'");

        for (int i = 0; i < filenames.length; i++) {
            TimeSeries series = new TimeSeries(filenames[i], inputNames, outputNames);

            int timeSeriesLength = series.getLength();
            Log.info("created a timeSeries with length: " + timeSeriesLength);
            if (timeSeriesLength > maxTimeSeriesLength) maxTimeSeriesLength = timeSeriesLength;

            timeSeries.add(series);
        }

        Log.info("max timeSeries length of the data set was: " + maxTimeSeriesLength);
    }

    /**
     * Returns the minimum value across all TimeSeries values for
     * each column.
     *
     * @return an array of the minimum values for each time series column across all time series
     */
    public double[] getMins() {
        //TODO: You need to implement this for Programming Assignment 2 - Part 3
        //You need to calculate the min of each column
        double[] minimum = new double[this.timeSeries.get(0).values[0].length];
        for(int i = 0; i < minimum.length; i++) {
            minimum[i] = Double.POSITIVE_INFINITY;
        }
        for(TimeSeries series : this.timeSeries) {
            for(int timeStep = 0; timeStep < series.values.length; timeStep++) {
                for(int col = 0; col < series.values[0].length; col++) {
                    if(series.values[timeStep][col] < minimum[col]) {
                        minimum[col] = series.values[timeStep][col];
                    }
                }
            }
        }
        return minimum;
    }

    /**
     * Returns the maximum value across all TimeSeries values for
     * each column.
     *
     * @return an array of the minimum values for each time series column across all time series
     */
    public double[] getMaxs() {
        //TODO: You need to implement this for Programming Assignment 2 - Part 3
        //You need to calculate the max of each column
        double[] maximum = new double[this.timeSeries.get(0).values[0].length];
        for(int i = 0; i < maximum.length; i++) {
            maximum[i] = Double.NEGATIVE_INFINITY;
        }
        for(TimeSeries series : this.timeSeries) {
            for(int timeStep = 0; timeStep < series.values.length; timeStep++) {
                for(int col = 0; col < series.values[0].length; col++) {
                    if(series.values[timeStep][col] > maximum[col]) {
                        maximum[col] = series.values[timeStep][col];
                    }
                }
            }
        }
        return maximum;
    }

    /*
     * Normalizes the data between the given minimum and maximum values for each column
     *
     */
    public void normalizeMinMax(double[] mins, double[] maxs) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //subtract the min value for each colum and then divide by the max minus the min
        for(TimeSeries series : this.timeSeries){
            series.normalizeMinMax(mins, maxs);
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
     * Gets the length of the longest timeSeries in the timeSeries
     *
     * @return the length of the longest timeSeries in the timeSeries
     */
    public int getMaxLength() {
        return maxTimeSeriesLength;
    }


    /**
     * Gets the number of timeSeries in this DataSet
     *
     * @return the number of timeSeries in this DataSet
     */
    public int getNumberSequences() {
        return timeSeries.size();
    }

    /**
     * Gets the number of values in the input data in the timeSeries
     * of this DataSet
     *
     * @return the number of inputs per timeSeries - this will always be 1 for character prediction
     */
    public int getNumberInputs() {
        return inputNames.length;
    }

    /**
     * Gets the number of values in the output data in the timeSeries
     * of this DataSet
     *
     * @return the number of outputs per timeSeries - this will always be 1 for character prediction
     */
    public int getNumberOutputs() {
        return outputNames.length;
    }


    /**
     * This randomly shuffles the orders of the timeSeries in the 
     * timeSeries ArrayList. This will be useful when we are implementing 
     * different versions of stochastic backpropagation.
     */
    public void shuffle() {
        Log.trace("Shuffling '" + name + "'");
        Collections.shuffle(timeSeries);
    }

    /**
     * This gets the timeSeries in the specified position in the
     * timeSeries ArrayList.
     *
     * @param position is the position of the timeSeries to retrieve
     * from the timeSeries ArrayList. position should be >= 0 and <
     * timeSeries.size() or else it will throw an ArrayOutOfBounds 
     * exception.
     */
    public Sequence getSequence(int position) {
        Log.trace("Getting timeSeries[" + position + "] from '" + name + "'");
        return timeSeries.get(position);
    }

    /**
     * This gets a consecutive set of timeSeries from the timeSeries
     * ArrayList. position should be >= 0 and numberOfTimeSeries should
     * be >= 1.
     *
     * @param position the position of the first timeSeries to return
     * @param numberOfTimeSeries is how many timeSeries to return. If 
     * position + numberOfTimeSeries is > than timeSeries.size() it will
     * return the remaining timeSeries in the timeSeries ArrayList.
     *
     * @return An ArrayList of the timeSeries specified by position and
     * numberOfTimeSeries. Its size will be <= numberOfTimeSeries.
     */
    public List<Sequence> getSequences(int position, int numberOfTimeSeries) {
        int endIndex = position + numberOfTimeSeries;
        if (endIndex > timeSeries.size()) endIndex = timeSeries.size();

        Log.trace("Getting timeSeries[" + position + " to " + endIndex + "] from '" + name + "'");
            
        List<TimeSeries> subList = timeSeries.subList(position, endIndex);

        List<Sequence> castedList = new ArrayList<Sequence>();
        for (int i = 0; i < subList.size(); i++) {
            castedList.add(subList.get(i));
        }
        return castedList;
    }

    /**
     * This gets the entire set of timeSeries from the timeSeries
     * ArrayList. 
     *
     * @return The ArrayList of the timeSeries of this data set.
     */
    public List<Sequence> getSequences() {
        List<Sequence> castedList = new ArrayList<Sequence>();
        for (int i = 0; i < timeSeries.size(); i++) {
            castedList.add(timeSeries.get(i));
        }
        return castedList;
    }
}
