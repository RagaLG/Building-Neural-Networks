/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class StepSizeTest {

    public static void main(String[] arguments) {
        //creating this data set should work correctly if the previous test passed.
        DataSet xorData = new DataSet("xor data", "./datasets/xor.txt");

        //test different step sizes for the small
        //neural network
        testNNStepSizes(xorData, PA11Tests.createSmallNeuralNetwork(xorData, LossFunction.L2_NORM), "smallNN");

        //test different step sizes for the small
        //neural network
        testNNStepSizes(xorData, PA11Tests.createLargeNeuralNetwork(xorData, LossFunction.L2_NORM), "largeNN");
     }

    /**
     * This tests the effets of the step size (learning rate)
     * on the small fully connected neural network generated
     * by PA11Tests.createSmallNeuralNetwork()
     */
    public static void testNNStepSizes(DataSet xorData, NeuralNetwork nn, String networkName) {
        try {
            //test using the 4 possible XOR instances (Batch Gradient Descent)
            List<Instance> instances = xorData.getInstances(0, 4);

            double[] initialWeights = new double[nn.getNumberWeights()];

            for (int i = 0; i < initialWeights.length; i++) {
                //give the test weights a spread of positive and negative values
                initialWeights[i] = (Math.random() * 2.0) - 1.0;
            }

            double stepSize = 1e-10;

            nn.setWeights(initialWeights);
            double[] gradient = nn.getGradient(instances);

            double initialOutput = nn.forwardPass(instances);
            Log.info("initial output:  size:                           " + String.format("%22.15f", initialOutput));

            double[] testWeights = new double[initialWeights.length];
            for (int i = 0; i < 15; i++) {
                testWeights = Vector.subtractVector(initialWeights, Vector.multiply(stepSize, gradient));

                nn.setWeights(testWeights);
                double stepOutput = nn.forwardPass(instances);

                Log.info("output with step size: " + String.format("%22.15f", stepSize) + " -- " + String.format("%22.15f", stepOutput) + ", difference from initial: " + String.format("%22.15f", (initialOutput - stepOutput)));

                stepSize *= 10;
            }

        } catch (Exception e) {
            Log.fatal("Failed testNStepSizes with " + networkName);
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


}
