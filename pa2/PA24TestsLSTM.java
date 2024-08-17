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
import network.LSTMNode;
import network.NodeType;
import network.RecurrentNeuralNetwork;
import network.RNNNodeType;
import network.NeuralNetworkException;

import util.Log;



public class PA24TestsLSTM {
    public static final int NUMBER_REPEATS = 1;
    public static final boolean generatingTestValues = false; //do not modify or it will overwrite my correctly generated test values

    public static void main(String[] arguments) {
        try {
            //test an LSTM node with one time step to make sure
            //a basic forward and backward pass is working (without data 
            //from previous or future time steps
            testLSTMForwardPass(3253 /*seed*/, 1 /*maxSequenceLength*/);

            //test an LSTM node with 5 time steps to make sure that
            //the forward and backward pass is working for when there
            //are multiple time steps
            testLSTMForwardPass(9283 /*seed*/, 5 /*maxSequenceLength*/);

            //test an LSTM node with 5 time steps to make sure that
            //the forward and backward pass is working for when there
            //are multiple time steps
            testLSTMForwardPass(12323 /*seed*/, 10 /*maxSequenceLength*/);
        } catch (NeuralNetworkException e) {
            System.err.println("LSTM tests failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }

        //now that we've tested the LSTM cell itself lets make sure it works
        //correctly inside of an RNN

        TimeSeriesDataSet dataSet = new TimeSeriesDataSet("flights data set",
                //new String[]{"./datasets/flight_0.csv", "./datasets/flight_1.csv", "./datasets/flight_2.csv", "./datasets/flight_3.csv"}, /* input file names */
                new String[]{"./datasets/flight_0_short.csv", "./datasets/flight_1_short.csv", "./datasets/flight_2_short.csv", "./datasets/flight_3_short.csv"}, /* inp    ut file names */
                new String[]{"AltAGL", "E1 RPM", "IAS", "LatAc", "NormAc", "Pitch", "Roll"}, /*parameter names for RNN input values */
                new String[]{"Pitch", "Roll"} /*parameter names for RNN target output values */
                );

        double[] mins = dataSet.getMins();
        double[] maxs = dataSet.getMaxs();

        Log.info("Data set had the following column mins: " + Arrays.toString(mins));
        Log.info("Data set had the following column maxs: " + Arrays.toString(maxs));

        //don't uncomment these as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray(mins, "pa24_mins", 0, 0);
        if (generatingTestValues) TestValues.writeArray(maxs, "pa24_maxs", 0, 0);

        try {
            Log.info("Checking normalization column mins");
            TestValues.testArray(mins, TestValues.readArray("pa24_mins", 0, 0), "pa24_mins", 0, 0);
            Log.info("normalization mins were correct.");

            Log.info("Checking normalization column maxs");
            TestValues.testArray(maxs, TestValues.readArray("pa24_maxs", 0, 0), "pa24_maxs", 0, 0);
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
        PA23Tests.testOneLayerBackwardPass("oneLayerLSTMTest", dataSet, RNNNodeType.LSTM, "feed forward", LossFunction.NONE, 12345, 0);
        PA23Tests.testOneLayerBackwardPass("oneLayerLSTMTest", dataSet, RNNNodeType.LSTM, "feed forward", LossFunction.L2_NORM, 12345, 0);
        PA23Tests.testOneLayerBackwardPass("oneLayerLSTMTest", dataSet, RNNNodeType.LSTM, "jordan", LossFunction.L2_NORM, 13231, 2);
        PA23Tests.testOneLayerBackwardPass("oneLayerLSTMTest", dataSet, RNNNodeType.LSTM, "elman", LossFunction.L1_NORM, 19823, 1);

        PA23Tests.testTwoLayerBackwardPass("twoLayerLSTMTest", dataSet, RNNNodeType.LSTM, "feed forward", LossFunction.NONE, 18323, 0);
        PA23Tests.testTwoLayerBackwardPass("twoLayerLSTMTest", dataSet, RNNNodeType.LSTM, "feed forward", LossFunction.L1_NORM, 18323, 0);
        PA23Tests.testTwoLayerBackwardPass("twoLayerLSTMTest", dataSet, RNNNodeType.LSTM, "jordan", LossFunction.L1_NORM, 25142, 2);
        PA23Tests.testTwoLayerBackwardPass("twoLayerLSTMTest", dataSet, RNNNodeType.LSTM, "elman", LossFunction.L2_NORM, 2918382, 0);
    }

    public static void testLSTMForwardPass(int seed, int maxSequenceLength) throws NeuralNetworkException {
        int layer = 1;
        int number = 1;
        LSTMNode lstmNode = new LSTMNode(layer, number, NodeType.HIDDEN, maxSequenceLength);

        Random generator = new Random(seed);
        double[] weights = new double[11];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (generator.nextDouble() * 0.10) - 0.05;
        }

        lstmNode.setWeights(0, weights);

        //save the input values so we can use them to calculate the numeric gradient for the LSTM node
        double[] inputValues = new double[maxSequenceLength];
        for (int i = 0; i < maxSequenceLength; i++) {
            inputValues[i] = generator.nextDouble();

            lstmNode.preActivationValue[i] = inputValues[i];
        }

        for (int i = 0; i < maxSequenceLength; i++) {
            lstmNode.propagateForward(i);

            Log.debug("lstmNode time step " + i);
            Log.debug("\tlstmNode.preActivationValue[" + i + "]: " + lstmNode.preActivationValue[i]);
            Log.debug("\tlstmNode.postActivationValue[" + i + "]: " + lstmNode.postActivationValue[i]);
            Log.debug("\tlstmNode.ct[" + i + "]: " + lstmNode.ct[i]);
            Log.debug("\tlstmNode.C[" + i + "]: " + lstmNode.C[i]);
            Log.debug("\tlstmNode.ft[" + i + "]: " + lstmNode.ft[i]);
            Log.debug("\tlstmNode.it[" + i + "]: " + lstmNode.it[i]);
            Log.debug("\tlstmNode.ot[" + i + "]: " + lstmNode.ot[i]);

            //do not uncomment these as they will overwrite the correct values I've generated for the test
            if (generatingTestValues) TestValues.writeValue(lstmNode.preActivationValue[i], "preActivationValue", seed, i, maxSequenceLength);
            if (generatingTestValues) TestValues.writeValue(lstmNode.postActivationValue[i], "postActivationValue", seed, i, maxSequenceLength);
            if (generatingTestValues) TestValues.writeValue(lstmNode.ct[i], "ct", seed, i, maxSequenceLength);
            if (generatingTestValues) TestValues.writeValue(lstmNode.C[i], "C", seed, i, maxSequenceLength);
            if (generatingTestValues) TestValues.writeValue(lstmNode.ft[i], "ft", seed, i, maxSequenceLength);
            if (generatingTestValues) TestValues.writeValue(lstmNode.it[i], "it", seed, i, maxSequenceLength);
            if (generatingTestValues) TestValues.writeValue(lstmNode.ot[i], "ot", seed, i, maxSequenceLength);

            Log.info("Checking preActivationValue for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
            TestValues.testValue(lstmNode.preActivationValue[i], TestValues.readValue("preActivationValue", seed, i, maxSequenceLength), "preActivationValue", seed, i, maxSequenceLength);

            Log.info("Checking postActivationValue for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
            TestValues.testValue(lstmNode.postActivationValue[i], TestValues.readValue("postActivationValue", seed, i, maxSequenceLength), "postActivationValue", seed, i, maxSequenceLength);

            Log.info("Checking c for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
            TestValues.testValue(lstmNode.ct[i], TestValues.readValue("ct", seed, i, maxSequenceLength), "ct", seed, i, maxSequenceLength);

            Log.info("Checking C for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
            TestValues.testValue(lstmNode.C[i], TestValues.readValue("C", seed, i, maxSequenceLength), "C", seed, i, maxSequenceLength);

            Log.info("Checking ft for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
            TestValues.testValue(lstmNode.ft[i], TestValues.readValue("ft", seed, i, maxSequenceLength), "ft", seed, i, maxSequenceLength);

            Log.info("Checking it for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
            TestValues.testValue(lstmNode.it[i], TestValues.readValue("it", seed, i, maxSequenceLength), "it", seed, i, maxSequenceLength);

            Log.info("Checking ot for seed " + seed + ", sequenceIndex " + i + ", and maxSequenceLength: " + maxSequenceLength);
            TestValues.testValue(lstmNode.ot[i], TestValues.readValue("ot", seed, i, maxSequenceLength), "ot", seed, i, maxSequenceLength);
        }


        double[] numericGradient = getLSTMNumericGradient(lstmNode, inputValues, weights, maxSequenceLength);
        Log.info("numeric gradient: " + Arrays.toString(numericGradient));

        //don't uncomment this as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray(numericGradient, "numeric_gradient", seed, maxSequenceLength);

        Log.info("Checking numeric_gradient for seed " + seed + ", and maxSequenceLength: " + maxSequenceLength);
        TestValues.testArray(numericGradient, TestValues.readArray("numeric_gradient", seed, maxSequenceLength), "numeric_gradient", seed, maxSequenceLength);

        getLSTMOutput(lstmNode, inputValues, weights, maxSequenceLength);

        for (int i = 0; i < maxSequenceLength; i++ ) {
            //set the deltas to pseudo-random outputs so we can
            //test the backwards pass
            lstmNode.delta[i] = 1.0;
        }

        for (int i = maxSequenceLength - 1; i >= 0; i--) {
            lstmNode.propagateBackward(i);
        }

        double[] deltas = new double[11];
        lstmNode.getDeltas(0, deltas);
        Log.debug("delta_wi: " + lstmNode.delta_wi);
        Log.debug("delta_wf: " + lstmNode.delta_wf);
        Log.debug("delta_wc: " + lstmNode.delta_wc);
        Log.debug("delta_wo: " + lstmNode.delta_wo);

        Log.debug("delta_ui: " + lstmNode.delta_ui);
        Log.debug("delta_uf: " + lstmNode.delta_uf);
        Log.debug("delta_uo: " + lstmNode.delta_uo);

        Log.debug("delta_bi: " + lstmNode.delta_bi);
        Log.debug("delta_bf: " + lstmNode.delta_bf);
        Log.debug("delta_bc: " + lstmNode.delta_bc);
        Log.debug("delta_bo: " + lstmNode.delta_bo);


        for (int j = 0; j < deltas.length; j++) {
            Log.debug("lstmNode.deltas[" + j + "]: " + deltas[j]);
        }

        String[] weightNames = new String[11];
        lstmNode.getWeightNames(0, weightNames);


        Log.info("checking to see if numeric gradient and backprop deltas are close enough.");
        if (!BasicTests.gradientsCloseEnough(numericGradient, deltas, weightNames)) {
            throw new NeuralNetworkException("backprop vs numeric gradient check failed for seed " + seed + " and maxSequenceLength" + maxSequenceLength);
        }
    }

    public static double getLSTMOutput(LSTMNode lstmNode, double[] inputs, double[] weights, int maxSequenceLength) {
        lstmNode.reset();
        lstmNode.setWeights(0, weights);

        for (int i = 0; i < maxSequenceLength; i++) {
            lstmNode.preActivationValue[i] = inputs[i];
        }

        for (int i = 0; i < maxSequenceLength; i++) {
            lstmNode.propagateForward(i);
        }

        double outputSum = 0.0;
        for (int i = 0; i < maxSequenceLength; i++) {
            outputSum += lstmNode.postActivationValue[i];
        }

        return outputSum;
    }

    public static double[] getLSTMNumericGradient(LSTMNode lstmNode, double[] inputs, double[] weights, int maxSequenceLength) {
        double[] numericGradient = new double[weights.length];
        double[] testWeights = new double[weights.length];

        double H = 0.0000001;
        for (int i = 0; i < numericGradient.length; i++) {
            System.arraycopy(weights, 0, testWeights, 0, weights.length);

            testWeights[i] = weights[i] + H;
            double error1 = getLSTMOutput(lstmNode, inputs, testWeights, maxSequenceLength);

            testWeights[i] = weights[i] - H;
            double error2 = getLSTMOutput(lstmNode, inputs, testWeights, maxSequenceLength);

            numericGradient[i] = (error1 - error2) / (2.0 * H);

            Log.trace("numericGradient[" + i + "]: " + numericGradient[i] + ", error1: " + error1 + ", error2: " + error2 + ", testWeight1: " + (weights[i] + H) + ", testWeight2: "     + (weights[i] - H));
        }

        return numericGradient;
    }
}

