/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.Random;

import data.Sequence;
import data.TimeSeriesDataSet;
import data.TimeSeries;

import network.LossFunction;
import network.RecurrentNeuralNetwork;
import network.RNNNodeType;
import network.NeuralNetworkException;

import util.Color;
import util.Log;



public class PA23Tests {
    public static final int NUMBER_REPEATS = 1;
    public static final boolean generatingTestValues = false; //do not modify or it will overwrite my correctly generated test values

    public static void main(String[] arguments) {
        TimeSeriesDataSet dataSet = new TimeSeriesDataSet("flights data set", 
                //new String[]{"./datasets/flight_0.csv", "./datasets/flight_1.csv", "./datasets/flight_2.csv", "./datasets/flight_3.csv"}, /* input file names */
                new String[]{"./datasets/flight_0_short.csv", "./datasets/flight_1_short.csv", "./datasets/flight_2_short.csv", "./datasets/flight_3_short.csv"}, /* input file names */
                new String[]{"AltAGL", "E1 RPM", "IAS", "LatAc", "NormAc", "Pitch", "Roll"}, /*parameter names for RNN input values */
                new String[]{"Pitch", "Roll"} /*parameter names for RNN target output values */
                );

        double[] mins = dataSet.getMins();
        double[] maxs = dataSet.getMaxs();

        Log.info("Data set had the following column mins: " + Arrays.toString(mins));
        Log.info("Data set had the following column maxs: " + Arrays.toString(maxs));

        //don't uncomment these as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray(mins, "pa23_mins", 0, 0);
        if (generatingTestValues) TestValues.writeArray(maxs, "pa23_maxs", 0, 0);

        try {
            Log.info("Checking normalization column mins");
            TestValues.testArray(mins, TestValues.readArray("pa23_mins", 0, 0), "pa23_mins", 0, 0);
            Log.info("normalization mins were correct.");

            Log.info("Checking normalization column maxs");
            TestValues.testArray(maxs, TestValues.readArray("pa23_maxs", 0, 0), "pa23_maxs", 0, 0);
            Log.info("normalization maxs were correct.");

        } catch (NeuralNetworkException e) {
            Log.fatal("Normalization not correctly implemented, calcualted the wrong normalization min and max values: " + e);
            e.printStackTrace();
            System.exit(1);
        }


        dataSet.normalizeMinMax(mins, maxs);
        Log.info("normalized the data");



        //test these with random initialization seeds and indexes
        //do not change these numbers as I've saved expected results files
        //for each of these
        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.NONE, 123485, 0);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.NONE, 1245, 2);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.NONE, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD L2_NORM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.L2_NORM, 123485, 0);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.L2_NORM, 1245, 2);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.L2_NORM, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER FEED FORWARD L1_NORM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.L1_NORM, 12485, 1);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.L1_NORM, 145, 3);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.L1_NORM, 13245, 0);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.NONE, 123485, 3);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.NONE, 1245, 1);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.NONE, 123245, 1);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN L2_NORM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.L2_NORM, 193485, 2);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.L2_NORM, 981245, 3);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.L2_NORM, 1232452, 0);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER JORDAN L1_NORM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.L1_NORM, 124385, 1);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.L1_NORM, 1435, 0);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.L1_NORM, 313245, 0);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.NONE, 123485, 0);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.NONE, 1245, 2);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.NONE, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN L2_NORM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.L2_NORM, 193485, 3);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.L2_NORM, 981245, 2);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.L2_NORM, 1232452, 1);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "ONE LAYER ELMAN L1_NORM TESTS" + Color.RESET);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.L1_NORM, 124385, 2);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.L1_NORM, 1435, 1);
        testOneLayerBackwardPass("oneLayerSeqTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.L1_NORM, 313245, 2);
        System.out.println("\n\n");


        System.out.println(Color.CYAN_BOLD_BRIGHT + "\n\n\nTWO LAYER TEST\n\n\n" + Color.RESET);

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.NONE, 123485, 1);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.NONE, 1245, 3);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.NONE, 123245, 2);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD L2_NORM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.L2_NORM, 123485, 3);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.L2_NORM, 1245, 1);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.L2_NORM, 123245, 0);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER FEED FORWARD L1_NORM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "feed forward", LossFunction.L1_NORM, 12485, 1);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "feed forward", LossFunction.L1_NORM, 145, 2);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "feed forward", LossFunction.L1_NORM, 13245, 2);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.NONE, 123485, 3);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.NONE, 1245, 2);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.NONE, 123245, 3);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN L2_NORM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.L2_NORM, 193485, 3);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.L2_NORM, 981245, 1);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.L2_NORM, 1232452, 2);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER JORDAN L1_NORM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "jordan", LossFunction.L1_NORM, 124385, 1);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "jordan", LossFunction.L1_NORM, 1435, 2);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "jordan", LossFunction.L1_NORM, 313245, 2);
        System.out.println("\n\n");


        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN NO LOSS FUNCTION TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.NONE, 123485, 3);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.NONE, 1245, 1);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.NONE, 123245, 2);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN L2_NORM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.L2_NORM, 193485, 0);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.L2_NORM, 981245, 3);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.L2_NORM, 1232452, 0);
        System.out.println("\n\n");

        System.out.println(Color.YELLOW_BOLD_BRIGHT + "TWO LAYER ELMAN L1_NORM TESTS" + Color.RESET);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.LINEAR, "elman", LossFunction.L1_NORM, 124385, 3);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.SIGMOID, "elman", LossFunction.L1_NORM, 1435, 2);
        testTwoLayerBackwardPass("twoLayerSeqTest", dataSet, RNNNodeType.TANH, "elman", LossFunction.L1_NORM, 313245, 1);
        System.out.println("\n\n");
    }

    public static void testOneLayerBackwardPass(String rnnName, TimeSeriesDataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction, int seed, int seriesIndex) {
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

            Sequence series = dataSet.getSequence(seriesIndex);
            //test the output values first
            double loss = rnn.forwardPass(series);

            double[][] outputValues = rnn.getOutputValues(series);
            //Don't uncomment these -- I use these to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.writeArray(outputValues, "output_values", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);
            if (generatingTestValues) TestValues.writeLoss(loss, networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);

            //check the output values
            Log.info("Checking output values for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            TestValues.testOutputValues(outputValues, TestValues.readArray("output_values", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex), networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);

            //check the loss function
            Log.info("Checking loss for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            TestValues.testLoss(loss, TestValues.readLoss(networkType, rnnName, nodeType, lossFunction, seed, seriesIndex), networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);


            Log.info("Checking numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            double[] numericGradient = rnn.getNumericGradient(series);
            //Don't uncomment this -- I use it to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.write1DArray(numericGradient, "numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);

            //Log.info("numericGradient: " + Arrays.toString(numericGradient));

            if (!BasicTests.gradientsCloseEnough(TestValues.read1DArray("numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex), numericGradient, rnn.getWeightNames())) {
                throw new NeuralNetworkException("Gradients not close enough on " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            }

            Log.info("Checking backprop vs numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberSequences(); i++) {
                    series = dataSet.getSequence(i);

                    for (int j = 0; j < weights.length; j++) {
                        //give the test weights some random positive and negative values
                        weights[j] = (generator.nextDouble() * 0.10) - 0.05;
                    }

                    rnn.setWeights(weights);

                    double[] backpropGradient = rnn.getGradient(series);
                    Log.info("Got backprop gradient!");

                    numericGradient = rnn.getNumericGradient(series);
                    Log.info("Got numeric gradient!");


                    if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient, rnn.getWeightNames())) {
                        throw new NeuralNetworkException("backprop vs numeric gradient check failed on repeat " + repeat + " and series " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
                    }

                    Log.info("Numeric gradients vs backprop passed repeat " + repeat + " and series " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
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
            Log.info(Color.GREEN_BOLD_BRIGHT + "PASSED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", seriesIndex: " + seriesIndex + Color.RESET);
        } else {
            Log.error("FAILED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", seriesIndex: " + seriesIndex);
        }
    }

    public static void testTwoLayerBackwardPass(String rnnName, TimeSeriesDataSet dataSet, RNNNodeType nodeType, String networkType, LossFunction lossFunction, int seed, int seriesIndex) {
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

            Sequence series = dataSet.getSequence(seriesIndex);
            //test the output values first
            double loss = rnn.forwardPass(series);

            double[][] outputValues = rnn.getOutputValues(series);
            //Don't uncomment these -- I use these to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.writeArray(outputValues, "output_values", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);
            if (generatingTestValues) TestValues.writeLoss(loss, networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);

            //check the output values
            Log.info("Checking output values for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            TestValues.testOutputValues(outputValues, TestValues.readArray("output_values", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex), networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);

            //check the loss function
            Log.info("Checking loss for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            TestValues.testLoss(loss, TestValues.readLoss(networkType, rnnName, nodeType, lossFunction, seed, seriesIndex), networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);

            Log.info("Checking numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            double[] numericGradient = numericGradient = rnn.getNumericGradient(series);
            //Don't uncomment this -- I use it to write out the example values from my working code
            //for you to test against, if you uncomment them you will overwrite my saved versions and
            //get wrong results
            if (generatingTestValues) TestValues.write1DArray(numericGradient, "numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex);

            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(TestValues.read1DArray("numeric_gradient", networkType, rnnName, nodeType, lossFunction, seed, seriesIndex), numericGradient, rnn.getWeightNames())) {
                throw new NeuralNetworkException("Gradients not close enough on " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            }

            Log.info("Checking backprop vs numeric gradients for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberSequences(); i++) {
                    series = dataSet.getSequence(i);

                    for (int j = 0; j < weights.length; j++) {
                        //give the test weights some random positive and negative values
                        weights[j] = (generator.nextDouble() * 0.10) - 0.05;
                    }

                    rnn.setWeights(weights);

                    double[] backpropGradient = rnn.getGradient(series);
                    Log.info("Got backprop gradient!");

                    numericGradient = rnn.getNumericGradient(series);
                    Log.info("Got numeric gradient!");


                    if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient, rnn.getWeightNames())) {
                        throw new NeuralNetworkException("backprop vs numeric gradient check failed on repeat " + repeat + " and series " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
                    }

                    Log.info("Numeric gradients vs backprop passed repeat " + repeat + " and series " + i + " for " + networkType + " " + rnnName + " " + nodeType.name() + " " + lossFunction.name() + " for seed " + seed + " and seriesIndex " + seriesIndex);
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
            Log.info(Color.GREEN_BOLD_BRIGHT + "PASSED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", seriesIndex: " + seriesIndex + Color.RESET);
        } else {
            Log.error("FAILED backward pass " + networkType + " " + rnnName + " on: " + dataSet.getName() + " with rnn node type: " + nodeType.name() + " and loss function: " + lossFunction.name() + ", seed: " + seed + ", seriesIndex: " + seriesIndex);
        }
    }
}

