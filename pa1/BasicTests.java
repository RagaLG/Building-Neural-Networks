/**
 * A helpful class of tests to make sure programming assignment 1
 * part 1 is working correctly
 */

import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;

public class BasicTests {

    public static void main(String[] arguments) {
        //This tests loading the xor.txt file. Any changes you
        //make should not break this.
        testLoadingXOR();

        //This tests creating a neural network. Any changes you
        //make should not break this.
        testXORNeuralNetwork();
    }


    /**
     * This provides a series of tests for the DataSet class when it
     * loads the xor.txt dataset. 
     */
    public static void testLoadingXOR() {
        boolean passed = true;

        Log.info("Loading the xor.txt file as a DataSet.");
        try {
            DataSet xorData = new DataSet("xor data", "./datasets/xor.txt");

            //Test getting each instance individually
            //make sure the 4 different instances from XOR were read correctly
            Instance i0 = xorData.getInstance(0);
            if (!i0.equals(new double[]{0}, new double[]{0, 0})) {
                Log.error("testLoadingXOR was not correct on getInstance 0.");
                Log.error("\tinstance was:     " + i0.toString());
                Log.error("\tshould have been: [0 : 0, 0]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstance 0.");
            }

            Instance i1 = xorData.getInstance(1);
            if (!i1.equals(new double[]{1}, new double[]{1, 0})) {
                Log.error("testLoadingXOR was not correct on getInstance 1.");
                Log.error("\tinstance was:     " + i1.toString());
                Log.error("\tshould have been: [1 : 1, 0]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstance 1.");
            }

            Instance i2 = xorData.getInstance(2);
            if (!i2.equals(new double[]{1}, new double[]{0, 1})) {
                Log.error("testLoadingXOR was not correct on getInstance 2.");
                Log.error("\tinstance was:     " + i2.toString());
                Log.error("\tshould have been: [1 : 0, 1]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstance 2.");
            }

            Instance i3 = xorData.getInstance(3);
            if (!i3.equals(new double[]{0}, new double[]{1, 1})) {
                Log.error("testLoadingXOR was not correct on getInstance 3.");
                Log.error("\tinstance was:     " + i3.toString());
                Log.error("\tshould have been: [0 : 1, 1]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstance 3.");
            }

            //Gets all instances as once to test the getInstances Method
            List<Instance> is = xorData.getInstances(0,4);
            if (!is.get(0).equals(new double[]{0}, new double[]{0, 0})) {
                Log.error("testLoadingXOR was not correct on getInstances 0.");
                Log.error("\tinstance was:     " + i0.toString());
                Log.error("\tshould have been: [0 : 0, 0]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstances 0.");
            }

            if (!is.get(1).equals(new double[]{1}, new double[]{1, 0})) {
                Log.error("testLoadingXOR was not correct on getInstances 1.");
                Log.error("\tinstance was:     " + i1.toString());
                Log.error("\tshould have been: [0 : 1 , 0]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstances 1.");
            }

            if (!is.get(2).equals(new double[]{1}, new double[]{0, 1})) {
                Log.error("testLoadingXOR was not correct on getInstances 2.");
                Log.error("\tinstance was:     " + i2.toString());
                Log.error("\tshould have been: [1 : 0, 1]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstances 2.");
            }

            if (!is.get(3).equals(new double[]{0}, new double[]{1, 1})) {
                Log.error("testLoadingXOR was not correct on getInstances 3.");
                Log.error("\tinstance was:     " + i3.toString());
                Log.error("\tshould have been: [0 : 1, 1]");
                passed = false;
            } else {
                Log.trace("testLoadingXOR passed getInstances 3.");
            }
        } catch (Exception e) {
            Log.fatal("Exception occurred in testLoadingXOR: " + e.toString());
            e.printStackTrace();
            passed = false;
        }

        if (passed) Log.info("Passed testLoadingXOR.");
        else Log.fatal("FAILED testLoadingXOR!");
    }

    /**
     * Checks to make sure that the NeuralNetwork.getWeights() and NeuralNetwork.setWeights(double[]) 
     * methods work correctly. After the weights are set with setWeights they should be returned
     * identically from getWeights.
     *
     * @param network is the neural network to test
     * @param networkName is a friendly user readable name for the network
     *
     * @throws NeuralNetworkException if the weights were not the same or if there was a problem
     * with getWeights or setWeights
     */
    public static void checkGetSetWeights(NeuralNetwork network, String networkName) throws NeuralNetworkException {
        Log.debug("Testing get/set weights on neural network '" + networkName + "'");
        int numberWeights = network.getNumberWeights();
        double[] testWeights = new double[numberWeights];
        for (int i = 0; i < numberWeights; i++) {
            testWeights[i] = i;
        }

        network.setWeights(testWeights);

        double[] testWeights2 = network.getWeights();

        boolean passed = true;
        for (int i = 0; i < numberWeights; i++) {
            Log.trace("testWeights[" + i + "]: " + testWeights[i] + ", testWeights2[" + i + "]: " + testWeights2[i]);

            if (testWeights[i] != testWeights2[i]) {
                throw new NeuralNetworkException("Failed getSetWeights test on " + networkName 
                        + ", testWeights[" + i + "] was " + testWeights[i] + " and testWeights2[" + i + "] was " + testWeights2[i] + ".");
            }
        }
    }


    /**
     * Tests three different fully connected networks on the XOR dataset
     * to make sure basic functionality is working correctly.
     */
    public static void testXORNeuralNetwork() {
        boolean passed = true;

        //creating this data set should work correctly if the previous test passed.
        DataSet xorData = new DataSet("xor data", "./datasets/xor.txt");

        try {
            //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
            //in the XOR dataset) and one hidden layer with 3 nodes.
            NeuralNetwork xorNeuralNetwork1 = new NeuralNetwork(xorData.getNumberInputs(), new int[]{3}, xorData.getNumberOutputs(), LossFunction.NONE);
            //make this a fully connected network
            xorNeuralNetwork1.connectFully();

            int numberWeights = xorNeuralNetwork1.getNumberWeights();
            if (numberWeights != 12) {
                //this network should have 12 weights, 9 for the edges and 3 for the 3 hidden nodes
                passed = false;
                throw new NeuralNetworkException("Failed getNumberWeights on XOR Neural Network 1, returned " + numberWeights + " which should have been 29.");
            }

            //set the weights and then get the weights to make
            //sure the weights we get are the same and in the
            //same order as the weights we set
            checkGetSetWeights(xorNeuralNetwork1, "xorNeuralNetwork1");

            Log.info("Passed testXORNeuralNetwork 1");
        } catch (Exception e) {
            Log.fatal("Failed testXORNeuralNetwork on Neural Network 1");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
            passed = false;
        }

        try {
            //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
            //in the XOR dataset) and two hidden layers with 3 and 4 nodes.
            NeuralNetwork xorNeuralNetwork2 = new NeuralNetwork(xorData.getNumberInputs(), new int[]{3, 4}, xorData.getNumberOutputs(), LossFunction.NONE);
            //make this a fully connected network
            xorNeuralNetwork2.connectFully();

            int numberWeights = xorNeuralNetwork2.getNumberWeights();
            if (numberWeights != 29) {
                //this network should have 29 weights:
                //6 from the 2 input to the 3 in the first hidden layer
                //12 from the 3 on hidden layer 1 to the 4 on hidden layer 2
                //4 from the 4 on hidden layer 2 to the 1 on the output layer
                //7 for the 7 hidden node biases
                passed = false;
                throw new NeuralNetworkException("Failed getNumberWeights on XOR Neural Network 2, returned " + numberWeights + " which should have been 29.");
            }

            //set the weights and then get the weights to make
            //sure the weights we get are the same and in the
            //same order as the weights we set
            checkGetSetWeights(xorNeuralNetwork2, "xorNeuralNetwork2");

            Log.info("Passed testXORNeuralNetwork 2");
        } catch (Exception e) {
            Log.fatal("Failed testXORNeuralNetwork on Neural Network 2");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
            passed = false;
        }


        try {
            //create an NeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output (the number of outputs 
            //in the XOR dataset) and three hidden layers with 3, 2 and 4 nodes.
            NeuralNetwork xorNeuralNetwork3 = new NeuralNetwork(xorData.getNumberInputs(), new int[]{3, 2, 4}, xorData.getNumberOutputs(), LossFunction.NONE);
            //make this a fully connected network
            xorNeuralNetwork3.connectFully();

            int numberWeights = xorNeuralNetwork3.getNumberWeights();
            if (numberWeights != 33) {
                //this network should have 26 weights:
                //6 from the 2 input to the 3 in the first hidden layer
                //6 from the 3 on hidden layer 1 to the 2 on hidden layer 2
                //8 from the 2 on hidden layer 2 to the 4 on hidden layer 3
                //4 from the 4 on hidden layer 3 to the 1 on the output layer
                //9 for the 9 hidden node biases
                passed = false;
                throw new NeuralNetworkException("Failed getNumberWeights on XOR Neural Network 3, returned " + numberWeights + " which should have been 33.");
            }

            //set the weights and then get the weights to make
            //sure the weights we get are the same and in the
            //same order as the weights we set
            checkGetSetWeights(xorNeuralNetwork3, "xorNeuralNetwork3");

            Log.info("Passed testXORNeuralNetwork 3");
        } catch (Exception e) {
            Log.fatal("Failed testXORNeuralNetwork on Neural Network 3");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
            passed = false;
        }

        if (passed) Log.info("Passed testXORNeuralNetwork.");
        else Log.fatal("FAILED testXORNeuralNetwork!");
    }

    /**
     * When comparing the weights and outputs of a neural network they might be
     * slightly different when calculated by the numeric gradient and backprop. Also
     * different operating systems and java versions may have exp, log and pow functions
     * implemnented slightly different. Even further, the order of operations with double
     * precision values can slightly change outputs, so when checking equality to the tests
     * we just want to make sure it's close enough.
     */
    public static boolean closeEnough(double n1, double n2) {
        return Math.abs(n1 - n2) < 2e-6;
    }

    /**
     * Use the BasicTestscloseEnough(double, double) method over two
     * arrays to determine if all array values are close enough.
     */
    public static boolean vectorsCloseEnough(double[] v1, double[] v2) {
        for (int i = 0; i < v1.length; i++) {
            //if any pair of elements aren't close enough the arrays
            //aren't close enough
            if (!closeEnough(v1[i], v2[i])) return false;
        }

        //if we reached here both were close enough
        return true;
    }

    /**
     * When comparing gradients we want to use a relative strategy, as for example if the gradients are close to 10
     * and they are off by 1e-5 it is not very bad, but if the gradients are close to 1e-4
     * and off by 1e-5 then the difference is much more significant.
     *
     * In general a relativeError > 1e-2 is a problem.
     * 1e-2 >= relativeError >= 1e-4 is not very good and indicative of a problem.
     * 1e-4 >= relativeError is good if your objective has a kink in it, not so good otherwise
     * (for now we're using tanh and sigmoid so this is not so good)
     * 1e-7 >= relative error is good
     */
    public static boolean gradientsCloseEnough(double[] g1, double[] g2) {
           double relativeError = Vector.norm(Vector.subtractVector(g1, g2)) / Math.max(Vector.norm(g1), Vector.norm(g2));

           if (relativeError >= 1e-4) {
               Log.error("relativeError bad: " + relativeError);
               for (int i = 0; i < g1.length; i++) {
                   Log.error("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]));
               }

           } else if (relativeError >= 1e-5) {
               Log.warning("relativeError probably bad: " + relativeError);
               for (int i = 0; i < g1.length; i++) {
                   Log.trace("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]));
               }

           } else if (relativeError >= 1e-7) {
               Log.debug("relativeError might be bad: " + relativeError);
               for (int i = 0; i < g1.length; i++) {
                   Log.trace("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]));
               }
           }

           return relativeError <= 1e-5;
    }

}
