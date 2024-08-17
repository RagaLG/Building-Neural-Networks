/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.Random;

import data.DataSet;
import data.Sequence;
import data.SequenceDataSet;
import data.CharacterSequence;
import data.SequenceException;

import network.LossFunction;
import network.RecurrentNeuralNetwork;
import network.RNNNodeType;
import network.NeuralNetworkException;

import util.Color;
import util.Log;



public class PA21Tests {
    public static final int HIDDEN_LAYER_SIZE = 5;
    public static final boolean generatingTestValues = false; //do not modify or it will overwrite my correctly generated test values

    public static void main(String[] arguments) {
        SequenceDataSet dataSet = new SequenceDataSet("penntreebank train small", "./datasets/penntreebank_train_small.txt");

        createOneLayerRNN("oneLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM);
        createOneLayerRNN("oneLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM);
        createOneLayerRNN("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX);
        createOneLayerRNN("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX);

        createTwoLayerRNN("twoLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM);
        createTwoLayerRNN("twoLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM);
        createTwoLayerRNN("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX);
        createTwoLayerRNN("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX);


        //test these with random initialization seeds and indexes
        //do not change these numbers as I've saved expected results files
        //for each of these
        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.NONE, 1245, 13);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.NONE, 11245, 13);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.NONE, 12145, 13);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD SVM TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SVM, 232456, 15);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SVM, 223456, 14);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SVM, 234562, 17);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD SOFTMAX TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SOFTMAX, 13245, 3);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SOFTMAX, 312345, 4);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SOFTMAX, 123453, 5);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.NONE, 132431, 1);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.NONE, 413231, 2);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.NONE, 132314, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN SVM TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SVM, 3253, 7);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SVM, 3532, 9);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM, 325322, 11);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN SOFTMAX TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SOFTMAX, 132431, 29);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SOFTMAX, 1231, 15);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SOFTMAX, 1331, 12);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.NONE, 132491, 21);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.NONE, 13931, 22);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "elman", LossFunction.NONE, 13239, 23);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN SVM TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SVM, 32132, 24);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SVM, 31132, 25);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "elman", LossFunction.SVM, 32531, 26);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN SOFTMAX TESTS" + Color.RESET);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SOFTMAX, 222431, 34);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX, 13221, 35);
        testOneLayerForwardPass("oneLayerPenn", dataSet, RNNNodeType.TANH, "elman", LossFunction.SOFTMAX, 15233, 36);
        System.out.println("\n\n");


        System.out.println(Color.CYAN_BOLD_BRIGHT + "\n\nTWO LAYER TESTS:\n\n" + Color.RESET);

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.NONE, 1245, 13);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.NONE, 11245, 13);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.NONE, 12145, 13);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD SVM TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SVM, 232456, 15);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SVM, 223456, 14);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SVM, 234562, 17);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD SOFTMAX TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SOFTMAX, 13245, 3);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SOFTMAX, 312345, 4);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SOFTMAX, 123453, 5);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.NONE, 132431, 1);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.NONE, 413231, 2);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.NONE, 132314, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN SVM TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SVM, 3253, 7);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SVM, 3532, 9);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM, 325322, 11);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN SOFTMAX TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SOFTMAX, 132431, 29);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SOFTMAX, 1231, 15);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SOFTMAX, 1331, 12);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.NONE, 132491, 21);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.NONE, 13931, 22);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "elman", LossFunction.NONE, 13239, 23);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN SVM TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SVM, 32132, 24);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SVM, 31132, 25);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "elman", LossFunction.SVM, 32531, 26);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN SOFTMAX TESTS" + Color.RESET);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SOFTMAX, 222431, 34);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX, 13221, 35);
        testTwoLayerForwardPass("twoLayerPenn", dataSet, RNNNodeType.TANH, "elman", LossFunction.SOFTMAX, 15233, 36);
        System.out.println("\n\n");
    }

    /**
     * This creates a RecurrentNeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output 
     * (the number of outputs in the XOR dataset) and one hidden layer with 2 nodes.
     *
     * @param DataSet should be the xorDataset
     *
     * @return A NeuralNetwork of the above size
     */
    public static RecurrentNeuralNetwork createOneLayerRNN(String rnnName, DataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction) {
        RecurrentNeuralNetwork oneLayerRNN = new RecurrentNeuralNetwork(dataSet.getNumberInputs(), new int[]{HIDDEN_LAYER_SIZE}, dataSet.getNumberOutputs(), dataSet.getMaxLength(), nodeType, lossFunction);

        try {
            oneLayerRNN.connectFully();

            int timeSkip = 1; //just test with recurrent edges doing the usual 1 time step skip
            if (networkType.equals("jordan")) {
                oneLayerRNN.connectJordan(timeSkip);
            } else if (networkType.equals("elman")) {
                oneLayerRNN.connectElman(timeSkip);
            }

            //input to hidden 1 = (48 input nodes * 48 hidden nodes) = 2304
            //hidden 1 bias = 48
            //hidden 1 to output = (48 input nodes * 48 output nodes) 2304
            int nInputs = dataSet.getNumberInputs();
            int nOutputs = dataSet.getNumberOutputs();
            int expectedWeights;
            if (nodeType == RNNNodeType.LSTM) {
                expectedWeights = (nInputs * HIDDEN_LAYER_SIZE) /*input to hidden*/ + (11 * HIDDEN_LAYER_SIZE) /*LSTM hidden weights/bias*/ + HIDDEN_LAYER_SIZE * nOutputs /*hidden to output*/ + (11 * nOutputs); /*weights/biases for outputs*/
            } else {
                expectedWeights = (nInputs * HIDDEN_LAYER_SIZE) /*input to hidden*/ + HIDDEN_LAYER_SIZE /*hidden bias*/ + HIDDEN_LAYER_SIZE * nOutputs; /*hidden to output*/
            }

            if (networkType.equals("jordan")) {
                //there will be a recurrent edge weight from the output node to each
                //hidden node
                expectedWeights += nOutputs * HIDDEN_LAYER_SIZE;
            } else if (networkType.equals("elman")) {
                //there will be a recurrent edge weight from each hidden
                //node in a layer to each other hidden node in that layer
                expectedWeights += HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE;
            }
            Log.info("network had " + oneLayerRNN.getNumberWeights() + " weights");


            int numberWeights = oneLayerRNN.getNumberWeights();
            if (numberWeights != expectedWeights) {
                throw new NeuralNetworkException("Failed getNumberWeights on " + networkType + " " + rnnName + ", returned " + numberWeights + " which should have been " + expectedWeights + ".");
            }

            //set the weights and then get the weights to make
            //sure the weights we get are the same and in the
            //same order as the weights we set
            BasicTests.checkGetSetWeights(oneLayerRNN, rnnName);

            Log.debug("successfully created " + networkType + " " + rnnName);
        } catch (Exception e) {
            Log.fatal("Failed creating " + rnnName);
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }

        return oneLayerRNN;
    }

    /**
     * This creates a RecurrentNeuralNetwork with 2 inputs (the number of inputs in the XOR dataset), 1 output 
     * (the number of outputs in the XOR dataset) and two hidden layer with 2 nodes.
     *
     * @param DataSet should be the xorDataset
     *
     * @return A NeuralNetwork of the above size
     */
    public static RecurrentNeuralNetwork createTwoLayerRNN(String rnnName, DataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction) {
        RecurrentNeuralNetwork twoLayerRNN = new RecurrentNeuralNetwork(dataSet.getNumberInputs(), new int[]{HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE}, dataSet.getNumberOutputs(), dataSet.getMaxLength(), nodeType, lossFunction);
        try {
            twoLayerRNN.connectFully();

            int timeSkip = 1; //just test with recurrent edges doing the usual 1 time step skip
            if (networkType.equals("jordan")) {
                twoLayerRNN.connectJordan(timeSkip);
            } else if (networkType.equals("elman")) {
                twoLayerRNN.connectElman(timeSkip);
            }


            int nInputs = dataSet.getNumberInputs();
            int nOutputs = dataSet.getNumberOutputs();
            int expectedWeights;
            if (nodeType == RNNNodeType.LSTM) {
                expectedWeights = (nInputs * HIDDEN_LAYER_SIZE) /*input to hidden1*/ + (HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE) /*hidden1 to hidden2*/ + (11 * HIDDEN_LAYER_SIZE) /*hidden bias 1*/ + (11 * HIDDEN_LAYER_SIZE) /*hidden bias 2*/ + HIDDEN_LAYER_SIZE * nOutputs /*hidden2 to output*/ + (11 * nOutputs); /*weights for LSTM outputs*/
            } else {
                expectedWeights = (nInputs * HIDDEN_LAYER_SIZE) /*input to hidden1*/ + (HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE) /*hidden1 to hidden2*/ + HIDDEN_LAYER_SIZE /*hidden bias 1*/ + HIDDEN_LAYER_SIZE /*hidden bias 2*/ + HIDDEN_LAYER_SIZE * nOutputs; /*hidden2 to output*/
            }


            if (networkType.equals("jordan")) {
                //there will be a recurrent edge weight from the output node to each
                //hidden node
                expectedWeights += (nOutputs * HIDDEN_LAYER_SIZE) + (nOutputs * HIDDEN_LAYER_SIZE);
            } else if (networkType.equals("elman")) {
                //there will be a recurrent edge weight from each hidden
                //node in a layer to each other hidden node in that layer
                expectedWeights += (HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE) + (HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE);
            }
            Log.info("network had " + twoLayerRNN.getNumberWeights() + " weights");

            int numberWeights = twoLayerRNN.getNumberWeights();
            if (numberWeights != expectedWeights) {
                throw new NeuralNetworkException("Failed getNumberWeights on " + networkType + " " + rnnName + ", returned " + numberWeights + " which should have been " + expectedWeights + ".");
            }

            //set the weights and then get the weights to make
            //sure the weights we get are the same and in the
            //same order as the weights we set
            BasicTests.checkGetSetWeights(twoLayerRNN, rnnName);

            Log.debug("successfully created " + networkType + " " + rnnName);
        } catch (Exception e) {
            Log.fatal("Failed creating " + rnnName);
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }

        return twoLayerRNN;
    }

    public static void testOneLayerForwardPass(String rnnName, SequenceDataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction, int seed, int sequenceIndex) {
        boolean passed = true;
        RecurrentNeuralNetwork rnn = createOneLayerRNN(rnnName, dataSet, nodeType, networkType, lossFunction);

        try {
            //seed the generator so we always get the same random numbers,
            //this will make it easier to generate a lot of small random
            //weights which we can use to test the output and loss function
            Random generator = new Random(seed);

            //generate small random weights between -0.05 and 0.05
            double[] weights = new double[rnn.getNumberWeights()];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = (generator.nextDouble() * 0.10) - 0.05;
            }
            rnn.setWeights(weights);

            Sequence sequence = dataSet.getSequence(sequenceIndex);
            //test the output values first
            double loss = rnn.forwardPass(sequence);

            double[][] outputValues = rnn.getOutputValues(sequence);
            //Don't uncomment these -- I use these to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.writeArray(outputValues, "output_values", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);
            if (generatingTestValues) TestValues.writeLoss(loss, networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

            //check the output values
            Log.info("Checking output values for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            TestValues.testOutputValues(outputValues, TestValues.readArray("output_values", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex), networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

            //check the loss function
            Log.info("Checking loss for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            TestValues.testLoss(loss, TestValues.readLoss(networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex), networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

        } catch (NeuralNetworkException e) {
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
            passed = false;
        }

        if (passed) {
            Log.info(Color.GREEN_BOLD_BRIGHT + "PASSED " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex + Color.RESET);
        } else {
            Log.error("FAILED " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex);
        }
    }

    public static void testTwoLayerForwardPass(String rnnName, SequenceDataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction, int seed, int sequenceIndex) {
        boolean passed = true;
        RecurrentNeuralNetwork rnn = createTwoLayerRNN(rnnName, dataSet, nodeType, networkType, lossFunction);

        try {
            //seed the generator so we always get the same random numbers,
            //this will make it easier to generate a lot of small random
            //weights which we can use to test the output and loss function
            Random generator = new Random(seed);

            //generate small random weights between -0.05 and 0.05
            double[] weights = new double[rnn.getNumberWeights()];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = (generator.nextDouble() * 0.10) - 0.05;
            }
            rnn.setWeights(weights);

            Sequence sequence = dataSet.getSequence(sequenceIndex);
            //test the output values first
            double loss = rnn.forwardPass(sequence);

            double[][] outputValues = rnn.getOutputValues(sequence);
            //Don't uncomment these -- I use these to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.writeArray(outputValues, "output_values", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);
            if (generatingTestValues) TestValues.writeLoss(loss, networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

            //check the output values
            Log.info("Checking output values for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            TestValues.testOutputValues(outputValues, TestValues.readArray("output_values", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex), networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

            //check the loss function
            Log.info("Checking loss for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            TestValues.testLoss(loss, TestValues.readLoss(networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex), networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

        } catch (NeuralNetworkException e) {
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
            passed = false;
        }

        if (passed) {
            Log.info(Color.GREEN_BOLD_BRIGHT + "PASSED " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex + Color.RESET);
        } else {
            Log.error("FAILED " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex);
        }
    }
}

