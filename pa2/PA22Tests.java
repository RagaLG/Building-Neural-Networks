/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.Random;

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



public class PA22Tests {
    public static final int NUMBER_REPEATS = 1;
    public static final boolean generatingTestValues = false; //do not modify or it will overwrite my correctly generated test values

    public static void main(String[] arguments) {
        SequenceDataSet dataSet = new SequenceDataSet("sequence test set", "./datasets/sequence_test_set.txt");

        //test these with random initialization seeds and indexes
        //do not change these numbers as I've saved expected results files
        //for each of these

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.NONE, 123485, 0);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.NONE, 1245, 1);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.NONE, 123245, 2);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD SOFTMAX TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SOFTMAX, 123485, 3);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SOFTMAX, 1245, 4);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SOFTMAX, 123245, 5);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD SVM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SVM, 12485, 0);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SVM, 145, 1);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SVM, 13245, 2);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.NONE, 123485, 0);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.NONE, 1245, 2);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.NONE, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN SOFTMAX TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SOFTMAX, 193485, 3);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SOFTMAX, 981245, 5);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SOFTMAX, 1232452, 2);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN SVM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SVM, 124385, 4);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SVM, 1435, 5);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM, 313245, 1);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.NONE, 123485, 1);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.NONE, 1245, 2);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.NONE, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN SOFTMAX TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SOFTMAX, 193485, 3);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX, 981245, 2);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.SOFTMAX, 1232452, 1);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN SVM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SVM, 124385, 1);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SVM, 1435, 4);
        testOneLayerBackwardPass("oneLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.SVM, 313245, 5);
        System.out.println("\n\n");


        System.out.println(Color.CYAN_BOLD_BRIGHT + "\n\n\nTWO LAYER TEST\n\n\n" + Color.RESET);

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.NONE, 123485, 4);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.NONE, 1245, 2);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.NONE, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD SOFTMAX TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SOFTMAX, 123485, 3);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SOFTMAX, 1245, 2);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SOFTMAX, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD SVM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.SVM, 12485, 1);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.SVM, 145, 4);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.SVM, 13245, 5);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.NONE, 123485, 0);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.NONE, 1245, 2);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.NONE, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN SOFTMAX TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SOFTMAX, 193485, 5);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SOFTMAX, 981245, 4);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SOFTMAX, 1232452, 1);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN SVM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.SVM, 124385, 2);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.SVM, 1435, 1);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.SVM, 313245, 4);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.NONE, 123485, 3);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.NONE, 1245, 0);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.NONE, 123245, 5);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN SOFTMAX TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SOFTMAX, 193485, 0);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SOFTMAX, 981245, 2);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.SOFTMAX, 1232452, 4);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN SVM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.SVM, 124385, 5);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.SVM, 1435, 1);
        testTwoLayerBackwardPass("twoLayerTimeSeriesTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.SVM, 313245, 0);
        System.out.println("\n\n");
    }

    public static void testOneLayerBackwardPass(String rnnName, SequenceDataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction, int seed, int sequenceIndex) {
        boolean passed = true;
        RecurrentNeuralNetwork rnn = PA21Tests.createOneLayerRNN(rnnName, dataSet, nodeType, networkType, lossFunction);

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


            Log.info("Checking numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            double[] numericGradient = numericGradient = rnn.getNumericGradient(sequence);
            //Don't uncomment this -- I use it to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.write1DArray(numericGradient, "numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(TestValues.read1DArray("numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex), numericGradient, rnn.getWeightNames())) {
                throw new NeuralNetworkException("Gradients not close enough on " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            }

            Log.info("Checking backprop vs numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberSequences(); i++) {
                    sequence = dataSet.getSequence(i);

                    for (int j = 0; j < weights.length; j++) {
                        //give the test weights some random positive and negative values
                        weights[j] = (generator.nextDouble() * 0.10) - 0.05;
                    }

                    rnn.setWeights(weights);

                    double[] backpropGradient = rnn.getGradient(sequence);
                    Log.info("Got backprop gradient!");

                    numericGradient = rnn.getNumericGradient(sequence);
                    Log.info("Got numeric gradient!");


                    if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient, rnn.getWeightNames())) {
                        throw new NeuralNetworkException("backprop vs numeric gradient check failed on repeat " + repeat + " and sequence " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
                    }

                    Log.info("Numeric gradients vs backprop passed repeat " + repeat + " and sequence " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
                }

                if ((repeat % 10) == 0) {
                    Log.trace("Numeric gradients vs backprop passed repeat " + repeat + " completed.");
                }
            }


        } catch (NeuralNetworkException e) {
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
            passed = false;
        }

        if (passed) {
            Log.info(Color.GREEN_BOLD_BRIGHT + "PASSED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex + Color.RESET);
        } else {
            Log.error("FAILED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex);
        }
    }

    public static void testTwoLayerBackwardPass(String rnnName, SequenceDataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction, int seed, int sequenceIndex) {
        boolean passed = true;
        RecurrentNeuralNetwork rnn = PA21Tests.createTwoLayerRNN(rnnName, dataSet, nodeType, networkType, lossFunction);

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

            Log.info("Checking numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            double[] numericGradient = numericGradient = rnn.getNumericGradient(sequence);
            //Don't uncomment this -- I use it to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.write1DArray(numericGradient, "numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex);

            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(TestValues.read1DArray("numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, sequenceIndex), numericGradient, rnn.getWeightNames())) {
                throw new NeuralNetworkException("Gradients not close enough on " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            }

            Log.info("Checking backprop vs numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberSequences(); i++) {
                    sequence = dataSet.getSequence(i);

                    for (int j = 0; j < weights.length; j++) {
                        //give the test weights some random positive and negative values
                        weights[j] = (generator.nextDouble() * 0.10) - 0.05;
                    }

                    rnn.setWeights(weights);

                    double[] backpropGradient = rnn.getGradient(sequence);
                    Log.info("Got backprop gradient!");

                    numericGradient = rnn.getNumericGradient(sequence);
                    Log.info("Got numeric gradient!");


                    if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient, rnn.getWeightNames())) {
                        throw new NeuralNetworkException("backprop vs numeric gradient check failed on repeat " + repeat + " and sequence " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
                    }

                    Log.info("Numeric gradients vs backprop passed repeat " + repeat + " and sequence " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and sequenceIndex " + sequenceIndex);
                }

                if ((repeat % 10) == 0) {
                    Log.trace("Numeric gradients vs backprop passed repeat " + repeat + " completed.");
                }
            }


        } catch (NeuralNetworkException e) {
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
            passed = false;
        }

        if (passed) {
            Log.info(Color.GREEN_BOLD_BRIGHT + "PASSED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex + Color.RESET);
        } else {
            Log.error("FAILED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", sequenceIndex: " + sequenceIndex);
        }
    }
}

