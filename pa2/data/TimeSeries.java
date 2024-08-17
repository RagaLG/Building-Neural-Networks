package data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import java.util.Arrays;
import java.util.ArrayList;



import util.Log;

public class TimeSeries implements Sequence {

    //the names of the different columns in the time series file
    String[] columnNames;

    //the indexes of the columns used for inputs to the RNN
    int[] inputIndexes;

    //the indexes of the columns used for outputs to the RNN
    int[] outputIndexes;


    /**
     * The different values in the time series file
     */
    double[][] values;

    /**
     * Creates a new TimeSeries object from a file
     *
     * @param filename is the file to create the TimeSeries from
     * @param inputNames are the column names of the input parameters
     * @param outputNames are the column names of the output parameters
     *
     */
    public TimeSeries(String filename, String[] inputNames, String[] outputNames) {
        try {
            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(filename)));

            Log.trace("'" + filename + "' contents are:");

            //the first line is the column names
            String readLine = bufferedReader.readLine();
            columnNames = readLine.split(",");
            Log.info("Column names are: " + Arrays.toString(columnNames));

            inputIndexes = new int[inputNames.length];
            outputIndexes = new int[outputNames.length];

            //set the indexes to -1 so we can check if they weren't found
            for (int i = 0; i < inputNames.length; i++) inputIndexes[i] = -1;
            for (int i = 0; i < outputNames.length; i++) outputIndexes[i] = -1;

            for (int i = 0; i < columnNames.length; i++) {
                for (int j = 0; j < inputNames.length; j++) {
                    if (columnNames[i].equals(inputNames[j])) {
                        inputIndexes[j] = i;
                    }
                }

                for (int j = 0; j < outputNames.length; j++) {
                    if (columnNames[i].equals(outputNames[j])) {
                        outputIndexes[j] = i;
                    }
                }
            }

            Log.info("inputIndexes: " + Arrays.toString(inputIndexes));
            Log.info("outputIndexes: " + Arrays.toString(outputIndexes));

            boolean indexNotFound = false;
            for (int i = 0; i < inputNames.length; i++) {
                if (inputIndexes[i] == -1) {
                    Log.fatal("ERROR: could not find input column name: '" + inputNames[i]);
                    indexNotFound = true;
                }
            }

            for (int i = 0; i < outputNames.length; i++) {
                if (outputIndexes[i] == -1) {
                    Log.fatal("ERROR: could not find output column name: '" + outputNames[i]);
                    indexNotFound = true;
                }
            }

            if (indexNotFound) System.exit(1);

            ArrayList<String> valueLines = new ArrayList<String>();

            //read the file line by line and add them to the valueLines array
            //list so we know how large to make our value array
            while ((readLine = bufferedReader.readLine()) != null) {
                valueLines.add(readLine);
            }
            
            values = new double[valueLines.size()][columnNames.length];

            for (int i = 0; i < valueLines.size(); i++) {
                String line = valueLines.get(i);
                String[] valueStrings = line.split(",");
                
                //Log.info("split line: " + Arrays.toString(valueStrings) + ", columnNames.length: " + columnNames.length + ", values.length: " + values.length + ", values[" + i + "].length: " + values[i].length);
                for (int j = 0; j < columnNames.length; j++) {
                    values[i][j] = Double.parseDouble(valueStrings[j]);
                }
            }

        } catch (IOException e) {
            Log.fatal("ERROR opening DataSet file: '" + filename + "'");
            e.printStackTrace();
            System.exit(1);
        }
    }


    /**
     * Normalizes the values of this time series between 0 and 1 given
     * the provided column min and max values.
     *
     * @param mins the minimum value for each column to be used in normalization
     * @param maxs the maximum value for each column to be used in normalization
     */
    public void normalizeMinMax(double[] mins, double[] maxs) {
        for (int timeStep = 0; timeStep < values.length; timeStep++) {
            for (int k = 0; k < values[0].length; k++) {
                double value = values[timeStep][k];
                values[timeStep][k] = (value - mins[k]) / (maxs[k] - mins[k]);
            }
        }
    }

    /**
     * Returns the value at the given time step for the specified parameter
     *
     * @param timeStep the timeStep of the int value to return
     * @param index is the index of the parameter being retrieved
     *
     * @return the value from the csv file at the time step for the given index
     */
    public double valueAt(int timeStep, int index) {
        return values[timeStep][index];
    }

    /**
     * Returns the nth input value for a given time step.
     *
     * @param timeStep is the time step for the input value to return
     * @param inputIndex is the input column number to get. We can use this 
     * to look up which actual index it is at in the values array by using
     * inputIndexes
     *
     * @return the nth input value for a given time step.
     */
    public double getInputValue(int timeStep, int inputIndex) {
        return values[timeStep][inputIndexes[inputIndex]];
    }

    /**
     * Returns the nth output value for a given time step.
     *
     * @param timeStep is the time step for the output value to return
     * @param outputIndex is the output column number to get. We can use this 
     * to look up which actual index it is at in the values array by using
     * outputIndexes
     *
     * @return the nth output value for a given time step.
     */
    public double getOutputValue(int timeStep, int outputIndex) {
        return values[timeStep][outputIndexes[outputIndex]];
    }

    /**
     * @return the number of columns in the time series
     */
    public int getNumberColumns() {
        return values[0].length;
    }

    /**
     * @return the length of this sequence
     */
    public int getLength() {
        return values.length;
    }

}
