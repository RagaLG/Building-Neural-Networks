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


public class PA12Tests {
    /**
     * NUMBER_REPEATS is used to repeatedly test the gradient comparison
     * between the gradient calculated by the finite difference method
     * and the gradient calculated using backprop. We can repeatedly
     * do this with random waits to be quite sure we've implemented both
     * methods correctly.
     */
    public static final int NUMBER_REPEATS = 100;

    public static DataSet xorData = new DataSet("xor data", "./datasets/xor.txt");

    public static void main(String[] arguments) {
        //test the numeric gradient calculations on a small
        //neural network
        testTinyGradientNumeric();

        //test the numeric gradient calculations on a small
        //neural network
        testSmallGradientNumeric();

        //test the numeric gradient calculations on a large 
        //neural network
        testLargeGradientNumeric();


        //this tests calculation of of the gradient via
        //the backwards pass for the small fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights
        testTinyGradients(xorData, LossFunction.NONE);


        //this tests calculation of of the gradient via
        //the backwards pass for the small fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights
        testSmallGradients(xorData, LossFunction.NONE);

        //this tests calculation of of the gradient via
        //the backwards pass for the large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights
        testLargeGradients(xorData, LossFunction.NONE);

        //this tests calculation of of the gradient via
        //the backwards pass for the tiny fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights
        testTinyGradientsMultiInstance(xorData, LossFunction.NONE);


        //this tests calculation of of the gradient via
        //the backwards pass for the small fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights
        testSmallGradientsMultiInstance(xorData, LossFunction.NONE);

        //this tests calculation of of the gradient via
        //the backwards pass for the large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights
        testLargeGradientsMultiInstance(xorData, LossFunction.NONE);
     }

    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumeric() {
        try {
            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(xorData, LossFunction.NONE);

            //test the 4 possible XOR instances
            Instance instance00 = new Instance(new double[]{1.0}, new double[]{0.0, 0.0});
            Instance instance10 = new Instance(new double[]{0.0}, new double[]{1.0, 0.0});
            Instance instance01 = new Instance(new double[]{0.0}, new double[]{0.0, 1.0});
            Instance instance11 = new Instance(new double[]{1.0}, new double[]{1.0, 1.0});

            double[] weights = new double[tinyNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            tinyNN.setWeights(weights);
            
            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, -0.06250000073038109, 0.0, -0.0875000000233328, 0.0};
            double[] numericGradient = tinyNN.getNumericGradient(instance00);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 00!");
            }
            Log.info("passed testTinyGradientNumeric on instance00");

            calculatedGradient = new double[]{-0.06249522344070613, -0.0872749678082485, 0.0, 0.0, -0.06249522344070613, 0.0, -0.0872749678082485, -0.012488639566932136};
            numericGradient = tinyNN.getNumericGradient(instance10);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 10!");
            }
            Log.info("passed testTinyGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, -0.06245759021084041, -0.08550240071514281, -0.06245759021084041, 0.0, -0.08550240071514281, -0.03719600183416105};
            numericGradient = tinyNN.getNumericGradient(instance01);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 01!");
            }
            Log.info("passed testTinyGradientNumeric on instance01");

            calculatedGradient = new double[]{-0.06242549310808698, -0.08399108686329981, -0.06242549310808698, -0.08399108686329981, -0.06242549310808698, 0.0, -0.08399108686329981, -0.04928500607626063};
            numericGradient = tinyNN.getNumericGradient(instance11);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 11!");
            }
            Log.info("passed testTinyGradientNumeric on instance11");

        } catch (Exception e) {
            Log.fatal("Failed testTinyGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }



    /**
     * This tests calculation of the numeric gradient for
     * the small fully connected neural network generated
     * by PA11Tests.createSmallNeuralNetwork()
     */
    public static void testSmallGradientNumeric() {
        try {
            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(xorData, LossFunction.NONE);

            //test the 4 possible XOR instances
            Instance instance00 = new Instance(new double[]{1.0}, new double[]{0.0, 0.0});
            Instance instance10 = new Instance(new double[]{0.0}, new double[]{1.0, 0.0});
            Instance instance01 = new Instance(new double[]{0.0}, new double[]{0.0, 1.0});
            Instance instance11 = new Instance(new double[]{1.0}, new double[]{1.0, 1.0});

            double[] weights = new double[smallNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            smallNN.setWeights(weights);

            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723891304958, 0.0, -0.1508745262057687};
            double[] numericGradient = smallNN.getNumericGradient(instance00);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 00!");
            }
            Log.info("passed testSmallGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 6.106226635438361E-9, 0.0, 0.0, 0.0, 0.0, 0.0, 6.106226635438361E-9, 0.0, 6.106226635438361E-9, 6.106226635438361E-9, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723835793807, -1.609823385706477E-8, -0.147720963794562};
            numericGradient = smallNN.getNumericGradient(instance10);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 10!");
            }
            Log.info("passed testSmallGradientNumeric on instance10");

            //calculatedGradient = new double[]{0.0, 0.0, 1.27675647831893E-8, 0.0, 0.0, 0.0, 0.0, 1.27675647831893E-8, 0.0, 1.0547118733938987E-8, 1.0547118733938987E-8, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723780282655, -3.164135620181696E-8, -0.15087452565065718};
            calculatedGradient = new double[]{
                0.0,
                0.0,
                1.27675647831893E-8,
                0.0,
                0.0,
                0.0,
                0.0,
                1.27675647831893E-8,
                0.0,
                1.0547118733938987E-8,
                0.0,
                0.0,
                4.440892098500626E-9,
                0.0,
                -0.14291749173001023,
                0.0,
                -0.14593435015974876,
                -3.164135620181696E-8,
                -0.15087452565065718
            };
            numericGradient = smallNN.getNumericGradient(instance01);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 01!");
            }
            Log.info("passed testSmallGradientNumeric on instance01");

            calculatedGradient = new double[]{
                0.0,
                1.887379141862766E-8,
                1.887379141862766E-8,
                0.0,
                0.0,
                0.0,
                0.0,
                1.887379141862766E-8,
                0.0,
                1.8318679906315083E-8,
                0.0,
                0.0,
                7.2164496600635175E-9,
                0.0,
                -0.14291749117489871,
                0.0,
                -0.14593435015974876,
                -4.884981308350689E-8,
                -0.1477209632394505
            };
            numericGradient = smallNN.getNumericGradient(instance11);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 11!");
            }
            Log.info("passed testSmallGradientNumeric on instance11");

        } catch (Exception e) {
            Log.fatal("Failed testSmallGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    /**
     * This tests calculation of the numeric gradient for
     * the large fully connected neural network generated
     * by PA11Tests.createLargeNeuralNetwork()
     */
    public static void testLargeGradientNumeric() {
        try {
            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(xorData, LossFunction.NONE);

            //test the 4 possible XOR instances
            Instance instance00 = new Instance(new double[]{1.0}, new double[]{0.0, 0.0});
            Instance instance10 = new Instance(new double[]{0.0}, new double[]{1.0, 0.0});
            Instance instance01 = new Instance(new double[]{0.0}, new double[]{0.0, 1.0});
            Instance instance11 = new Instance(new double[]{1.0}, new double[]{1.0, 1.0});

            double[] weights = new double[largeNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            largeNN.setWeights(weights);

            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2322144032618212, 0.0, -0.2353438832702892, 0.0, -0.21879212330766507, 0.0, -0.2400798448931596};
            double[] numericGradient = largeNN.getNumericGradient(instance00);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 00!");
            }
            Log.info("passed testLargeGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2322144032618212, 0.0, -0.2353438832702892, 0.0, -0.22038990632466948, 0.0, -0.2400798448931596};
            numericGradient = largeNN.getNumericGradient(instance10);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 10!");
            }
            Log.info("passed testLargeGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2339974491949448, 0.0, -0.2353438832702892, 0.0, -0.22926425757852087, 0.0, -0.2400798504442747};
            numericGradient = largeNN.getNumericGradient(instance01);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 01!");
            }
            Log.info("passed testLargeGradientNumeric on instance01");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2339974491949448, 0.0, -0.2353438832702892, 0.0, -0.2304554880261378, -5.551115123125783E-9, -0.2400798504442747};
            numericGradient = largeNN.getNumericGradient(instance11);
            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 11!");
            }
            Log.info("passed testLargeGradientNumeric on instance11");


        } catch (Exception e) {
            Log.fatal("Failed testLargeGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }

    public static void testTinyGradients(DataSet dataSet, LossFunction lossFunction) {
        try {
            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(dataSet, lossFunction);

            //test all the XOR instances

            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                    Instance instance = dataSet.getInstance(i);

                    double[] weights = new double[tinyNN.getNumberWeights()];

                    for (int j = 0; j < weights.length; j++) {
                        //give the test weights some random positive and negative values
                        weights[j] = (Math.random() * 2.0) - 1.0;
                    }

                    tinyNN.setWeights(weights);

                    double[] numericGradient = tinyNN.getNumericGradient(instance);
                    double[] backpropGradient = tinyNN.getGradient(instance);
                    if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient)) {
                        throw new NeuralNetworkException("testTinyGradients failed on repeat " + repeat + " and instance" + i + "!");
                    }
                }
                Log.trace("testTinyGradients passed repeat " + repeat + "!");

                if ((repeat % 10) == 0) {
                    Log.info("testTinyGradients repeat " + repeat + " completed.");
                }
            }

        } catch (Exception e) {
            Log.fatal("Failed testTinyGradients");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }

    public static void testSmallGradients(DataSet dataSet, LossFunction lossFunction) {
        try {
            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(dataSet, lossFunction);

            //test all the XOR instances

            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                    Instance instance = dataSet.getInstance(i);

                    double[] weights = new double[smallNN.getNumberWeights()];

                    for (int j = 0; j < weights.length; j++) {
                        //give the test weights some random positive and negative values
                        weights[j] = (Math.random() * 2.0) - 1.0;
                    }

                    smallNN.setWeights(weights);

                    double[] numericGradient = smallNN.getNumericGradient(instance);
                    double[] backpropGradient = smallNN.getGradient(instance);
                    if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient)) {
                        throw new NeuralNetworkException("testSmallGradients failed on repeat " + repeat + " and instance" + i + "!");
                    }
                }
                Log.trace("testSmallGradients passed repeat " + repeat + "!");

                if ((repeat % 10) == 0) {
                    Log.info("testSmallGradients repeat " + repeat + " completed.");
                }
            }

        } catch (Exception e) {
            Log.fatal("Failed testSmallGradients");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    public static void testLargeGradients(DataSet dataSet, LossFunction lossFunction) {
        try {
            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(dataSet, lossFunction);

            //test all the XOR instances

            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                    Instance instance = dataSet.getInstance(i);

                    double[] weights = new double[largeNN.getNumberWeights()];

                    for (int j = 0; j < weights.length; j++) {
                        //give the test weights some random positive and negative values
                        weights[j] = (Math.random() * 2.0) - 1.0;
                    }

                    largeNN.setWeights(weights);

                    double[] numericGradient = largeNN.getNumericGradient(instance);
                    double[] backpropGradient = largeNN.getGradient(instance);

                    if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient)) {
                        throw new NeuralNetworkException("testLargeGradients failed on repeat " + repeat + " and instance" + i + "!");
                    }
                }
                Log.trace("testLargeGradients passed repeat " + repeat + "!");

                if ((repeat % 10) == 0) {
                    Log.info("testLargeGradients repeat " + repeat + " completed.");
                }
            }

        } catch (Exception e) {
            Log.fatal("Failed testLargeGradients");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }

    }

    public static void testNetworkOnInstances(NeuralNetwork nn, List<Instance> instances, String description) throws NeuralNetworkException {
        double[] weights = new double[nn.getNumberWeights()];

        for (int j = 0; j < weights.length; j++) {
            //give the test weights some random positive and negative values
            weights[j] = (Math.random() * 2.0) - 1.0;
        }

        nn.setWeights(weights);

        double[] numericGradient = nn.getNumericGradient(instances);
        double[] backpropGradient = nn.getGradient(instances);

        if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient)) {
            throw new NeuralNetworkException(description + " failed!");
        }

        Log.trace(description + " passed!");
    }

    public static void testTinyGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction) {
        try {
            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(dataSet, lossFunction);

            //test all the XOR instances

            List<Instance> instances = null;
            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                    instances = dataSet.getInstances(0, 2);
                    testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + repeat + ", instances 0, 2");

                    instances = dataSet.getInstances(1, 2);
                    testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + repeat + ", instances 1, 2");

                    instances = dataSet.getInstances(2, 2);
                    testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + repeat + ", instances 2, 2");

                    instances = dataSet.getInstances(0, 3);
                    testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + repeat + ", instances 0, 3");

                    instances = dataSet.getInstances(1, 3);
                    testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + repeat + ", instances 1, 3");

                    instances = dataSet.getInstances(0, 4);
                    testNetworkOnInstances(tinyNN, instances, "testTinyGradientsMultiInstance, repeat " + repeat + ", instances 0, 4");

                }

                if ((repeat % 10) == 0) {
                    Log.info("testTinyGradientsMultiInstance repeat " + repeat + " completed.");
                }
            }

        } catch (Exception e) {
            Log.fatal("Failed testTinyGradientsMultiInstance");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    public static void testSmallGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction) {
        try {
            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(dataSet, lossFunction);

            //test all the XOR instances

            List<Instance> instances = null;
            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                    instances = dataSet.getInstances(0, 2);
                    testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + repeat + ", instances 0, 2");

                    instances = dataSet.getInstances(1, 2);
                    testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + repeat + ", instances 1, 2");

                    instances = dataSet.getInstances(2, 2);
                    testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + repeat + ", instances 2, 2");

                    instances = dataSet.getInstances(0, 3);
                    testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + repeat + ", instances 0, 3");

                    instances = dataSet.getInstances(1, 3);
                    testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + repeat + ", instances 1, 3");

                    instances = dataSet.getInstances(0, 4);
                    testNetworkOnInstances(smallNN, instances, "testSmallGradientsMultiInstance, repeat " + repeat + ", instances 0, 4");

                }

                if ((repeat % 10) == 0) {
                    Log.info("testSmallGradientsMultiInstance repeat " + repeat + " completed.");
                }
            }

        } catch (Exception e) {
            Log.fatal("Failed testSmallGradientsMultiInstance");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    public static void testLargeGradientsMultiInstance(DataSet dataSet, LossFunction lossFunction) {
        try {
            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(dataSet, lossFunction);

            //test all the XOR instances
            List<Instance> instances = null;
            for (int repeat = 0; repeat < NUMBER_REPEATS; repeat++) {
                for (int i = 0; i < dataSet.getNumberInstances(); i++) {
                    instances = dataSet.getInstances(0, 2);
                    testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + repeat + ", instances 0, 2");

                    instances = dataSet.getInstances(1, 2);
                    testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + repeat + ", instances 1, 2");

                    instances = dataSet.getInstances(2, 2);
                    testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + repeat + ", instances 2, 2");

                    instances = dataSet.getInstances(0, 3);
                    testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + repeat + ", instances 0, 3");

                    instances = dataSet.getInstances(1, 3);
                    testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + repeat + ", instances 1, 3");

                    instances = dataSet.getInstances(0, 4);
                    testNetworkOnInstances(largeNN, instances, "testLargeGradientsMultiInstance, repeat " + repeat + ", instances 0, 4");

                }

                if ((repeat % 10) == 0) {
                    Log.info("testLargeGradientsMultiInstance repeat " + repeat + " completed.");
                }
            }

        } catch (Exception e) {
            Log.fatal("Failed testLargeGradientsMultiInstance");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }

    }
}

