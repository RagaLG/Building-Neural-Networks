/**
 * Use this to store the large arrays and loss function values to compare
 * correctness of forward passes
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.Arrays;
import java.util.List;

import data.SequenceDataSet;
import data.CharacterSequence;
import data.SequenceException;

import network.LossFunction;
import network.RecurrentNeuralNetwork;
import network.RNNNodeType;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class TestValues {

    public static double readValue(String name, int seed, int sequenceIndex, int maxSequenceLength) {
        String valueFilename = "./test_values/value_" + name + "_" + seed + "_" + sequenceIndex + "_" + maxSequenceLength;
        valueFilename = valueFilename.replace(' ', '_');

        Log.debug("Reading from value test file: '" + valueFilename + "'" );

        double value = -1;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(valueFilename)));

            String line = br.readLine();

            value = Double.parseDouble(line);
        } catch (IOException e) {
            Log.fatal("Error reading from value test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from value test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return value;
    }

    public static void writeValue(double value, String name, int seed, int sequenceIndex, int maxSequenceLength) {
        String valueFilename = "./test_values/value_" + name + "_" + seed + "_" + sequenceIndex + "_" + maxSequenceLength;
        valueFilename = valueFilename.replace(' ', '_');

        Log.debug("Writing to value test file: '" + valueFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(valueFilename)));

            br.write(String.valueOf(value));

        } catch (IOException e) {
            Log.fatal("Error reading from value test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from value test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
    }

    /**
     * Used to check if two values are close enough.
     */
    public static boolean testValue(double n1, double n2, String name, int seed, int sequenceIndex, int maxSequenceLength) throws NeuralNetworkException {
        if (Math.abs(n1 - n2) > 1e-8) {
            throw new NeuralNetworkException(name + " for seed " + seed + ", sequenceIndex " + sequenceIndex + " and maxSequenceLength: " + maxSequenceLength + " calculated value: " + n1 + " was not close enough to precomputed value: " + n2);
        } else {
            return true;
        }
    }

    public static void writeArray(double[] array, String arrayName, int seed, int maxSequenceLength) {
        String arrayFilename = "./test_values/" + arrayName + "_" + seed + "_" + maxSequenceLength;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Writing to " + arrayName + " test file: '" + arrayFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            br.write(String.valueOf(array.length));
            br.write("\n");

            //write each value of the array on its own line in order
            for (int i = 0; i < array.length; i++) {
                br.write(String.valueOf(array[i]));
                br.write("\n");
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
     }

    public static double[] readArray(String arrayName, int seed, int maxSequenceLength) {
        String arrayFilename = "./test_values/" + arrayName + "_" + seed + "_" + maxSequenceLength;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Reading from " + arrayName + " test file: '" + arrayFilename + "'" );

        double[] array = null;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            String line = br.readLine();
            int d1 = Integer.parseInt(line);
            //read each value of the array from its own line in order
            array = new double[d1];
            for (int i = 0; i < d1; i++) {
                line = br.readLine();
                array[i] = Double.parseDouble(line);
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return array;
    }

    /**
     * Compare if two arrays are close enough to be the same
     */
    public static boolean testArray(double[] m1, double[] m2, String arrayName, int seed, int maxSequenceLength) throws NeuralNetworkException {
        if (m1.length != m2.length) {
            throw new NeuralNetworkException(arrayName + " for seed " + seed + " and maxSequenceLength" + maxSequenceLength + " dimensions were not okay. Pre-computed was " + m2.length + " and calculated was : " + m1.length);
        }

        for (int i = 0; i < m1.length; i++) {
            if (Math.abs(m1[i] - m2[i]) > 1e-8) {
                throw new NeuralNetworkException(arrayName + " for seed " + seed + " and maxSequenceLength " + maxSequenceLength + " outputs were not the same at index[" + i + "] calculated: " + m1[i] + " vs. precomputed: " + m2[i]);
            }
        }

        return true;
    }










    public static double readLoss(String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) {
        String lossFilename = "./test_values/loss_" + networkType + "_" + rnnName + "_" + nodeType.name() + "_" + lossFunction.name() + "_" + seed + "_" + sequenceIndex;
        lossFilename = lossFilename.replace(' ', '_');

        Log.debug("Reading from loss test file: '" + lossFilename + "'" );

        double loss = -1;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(lossFilename)));

            String line = br.readLine();

            loss = Double.parseDouble(line);
        } catch (IOException e) {
            Log.fatal("Error reading from loss test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from loss test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return loss;
    }

    public static void writeLoss(double loss, String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) {
        String lossFilename = "./test_values/loss_" + networkType + "_" + rnnName + "_" + nodeType.name() + "_" + lossFunction.name() + "_" + seed + "_" + sequenceIndex;
        lossFilename = lossFilename.replace(' ', '_');

        Log.debug("Writing to loss test file: '" + lossFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(lossFilename)));

            br.write(String.valueOf(loss));

        } catch (IOException e) {
            Log.fatal("Error reading from loss test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from loss test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
    }

    public static double[][] readArray(String arrayName, String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) {
        String arrayFilename = "./test_values/" + arrayName + "_" + networkType + "_" + rnnName + "_" + nodeType.name() + "_" + lossFunction.name() + "_" + seed + "_" + sequenceIndex;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Reading from " + arrayName + " test file: '" + arrayFilename + "'" );

        double[][] array = null;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            String line = br.readLine();
            int d1 = Integer.parseInt(line);

            //the first line is the second dimension of the output values array
            line = br.readLine();
            int d2 = Integer.parseInt(line);

            //read each value of the array from its own line in order
            array = new double[d1][d2];
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    line = br.readLine();
                    array[i][j] = Double.parseDouble(line);
                }
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return array;
    }

    public static double[] read1DArray(String arrayName, String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) {
        String arrayFilename = "./test_values/" + arrayName + "_" + networkType + "_" + rnnName + "_" + nodeType.name() + "_" + lossFunction.name() + "_" + seed + "_" + sequenceIndex;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Reading from " + arrayName + " test file: '" + arrayFilename + "'" );

        double[] array = null;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            String line = br.readLine();
            int d1 = Integer.parseInt(line);
            //read each value of the array from its own line in order
            array = new double[d1];
            for (int i = 0; i < d1; i++) {
                line = br.readLine();
                array[i] = Double.parseDouble(line);
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return array;
    }


    public static void writeArray(double[][] array, String arrayName, String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) {
        String arrayFilename = "./test_values/" + arrayName + "_" + networkType + "_" + rnnName + "_" + nodeType.name() + "_" + lossFunction.name() + "_" + seed + "_" + sequenceIndex;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Writing to " + arrayName + " test file: '" + arrayFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            br.write(String.valueOf(array.length));
            br.write("\n");

            //the first line is the second dimension of the output values array
            br.write(String.valueOf(array[0].length));
            br.write("\n");

            //write each value of the array on its own line in order
            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    br.write(String.valueOf(array[i][j]));
                    br.write("\n");
                }
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
     }

    public static void write1DArray(double[] array, String arrayName, String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) {
        String arrayFilename = "./test_values/" + arrayName + "_" + networkType + "_" + rnnName + "_" + nodeType.name() + "_" + lossFunction.name() + "_" + seed + "_" + sequenceIndex;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Writing to " + arrayName + " test file: '" + arrayFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            br.write(String.valueOf(array.length));
            br.write("\n");

            //write each value of the array on its own line in order
            for (int i = 0; i < array.length; i++) {
                br.write(String.valueOf(array[i]));
                br.write("\n");
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
     }




    /**
     * Used to check if RNN loss and output values are close enough.
     */
    public static boolean testLoss(double n1, double n2, String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) throws NeuralNetworkException {
        if (Math.abs(n1 - n2) > 1e-8) {
            throw new NeuralNetworkException(networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex + " calculated loss: " + n1 + " was not close enough to precomputed loss: " + n2);
        } else {
            return true;
        }
    }

    /**
     * Use the BasicTestscloseEnough(double[], double[]) method over two
     * 2d arrays to determine if all array values are close enough.
     */
    public static boolean testOutputValues(double[][] m1, double[][] m2, String networkType, String rnnName, RNNNodeType nodeType, LossFunction lossFunction, int seed, int sequenceIndex) throws NeuralNetworkException {
        if (m1.length != m2.length || m1[0].length != m2[0].length) {
            throw new NeuralNetworkException(networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex + " weight dimensions were not okay. Pre-computed were " + m2.length + "x" + m2[0].length + " and calculated were: " + m1.length + "x" + m1[0].length);
        }

        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[i].length; j++) {
                if (Math.abs(m1[i][j] - m2[i][j]) > 1e-8) {
                    throw new NeuralNetworkException(networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex + " outputs were not the same for time step: " + i + " and output value: " + j + ", calculated: " + m1[i][j] + " vs. precomputed: " + m2[i][j]);
                }
            }
        }

        return true;
    }


}
