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


public class PA13Tests {
    public static DataSet xorData = new DataSet("xor data", "./datasets/xor.txt");

    public static void main(String[] arguments) {
        if (arguments.length != 1) {
            System.err.println("Invalid arguments, you must specify a loss function, usage: ");
            System.err.println("\tjava PA13Tests <loss function>");
            System.err.println("\tloss function options are: 'none', 'l1_norm' or 'l2_norm'");
            System.exit(1);
        }

        String lossFunctionName = arguments[0];

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("none")) {
            Log.info("Using an L1_NORM loss function.");
            lossFunction = LossFunction.NONE;

            //test the numeric gradient calculations on tiny,
            //small and large neural networks with no
            //loss function
            testTinyGradientNumericNone();
            testSmallGradientNumericNone();
            testLargeGradientNumericNone();


        } else if (lossFunctionName.equals("l1_norm")) {
            Log.info("Using an L1_NORM loss function.");
            lossFunction = LossFunction.L1_NORM;

            //test the numeric gradient calculations on tiny,
            //small and large neural networks with the L1_NORM
            //loss function
            testTinyGradientNumericL1();
            testSmallGradientNumericL1();
            testLargeGradientNumericL1();

        } else if (lossFunctionName.equals("l2_norm")) {
            Log.info("Using an L2_NORM loss function.");
            lossFunction = LossFunction.L2_NORM;

            //test the numeric gradient calculations on tiny,
            //small and large neural networks with the L2_NORM
            //loss function
            testTinyGradientNumericL2();
            testSmallGradientNumericL2();
            testLargeGradientNumericL2();

        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        //these tests calculation of of the gradient via
        //the backwards pass for the tiny, small and large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights when the network has a L2_NORM
        //loss function
        PA12Tests.testTinyGradients(xorData, lossFunction);
        PA12Tests.testSmallGradients(xorData, lossFunction);
        PA12Tests.testLargeGradients(xorData, lossFunction);

        //this tests calculation of of the gradient via
        //the backwards pass for the tiny, small and large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights when the network has a L2_NORM
        //loss function
        PA12Tests.testTinyGradientsMultiInstance(xorData, lossFunction);
        PA12Tests.testSmallGradientsMultiInstance(xorData, lossFunction);
        PA12Tests.testLargeGradientsMultiInstance(xorData, lossFunction);
     }

    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumericNone() {
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
            
            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, -0.06250000086915897, 0.0, -0.08749999988455492, 0.0};
            double[] numericGradient = tinyNN.getNumericGradient(instance00);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 00!");
            }
            Log.info("passed testTinyGradientNumeric on instance00");

            calculatedGradient = new double[]{-0.06249522344070613, -0.0872749678082485, 0.0, 0.0, -0.06249522344070613, 0.0, -0.0872749678082485, -0.012488639566932136};
            numericGradient = tinyNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 10!");
            }
            Log.info("passed testTinyGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, -0.06245759021084041, -0.08550240071514281, -0.06245759021084041, 0.0, -0.08550240071514281, -0.03719600183416105};
            numericGradient = tinyNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 01!");
            }
            Log.info("passed testTinyGradientNumeric on instance01");

            calculatedGradient = new double[]{-0.06242549310808698, -0.08399108686329981, -0.06242549310808698, -0.08399108686329981, -0.06242549310808698, 0.0, -0.08399108686329981, -0.04928500607626063};
            numericGradient = tinyNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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
    public static void testSmallGradientNumericNone() {
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
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 00!");
            }
            Log.info("passed testSmallGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 6.106226635438361E-9, 0.0, 0.0, 0.0, 0.0, 0.0, 6.106226635438361E-9, 0.0, 6.106226635438361E-9, 6.106226635438361E-9, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723835793807, -1.609823385706477E-8, -0.147720963794562};
            numericGradient = smallNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 10!");
            }
            Log.info("passed testSmallGradientNumeric on instance10");

            //calculatedGradient = new double[]{0.0, 0.0, 1.27675647831893E-8, 0.0, 0.0, 0.0, 0.0, 1.27675647831893E-8, 0.0, 1.0547118733938987E-8, 1.0547118733938987E-8, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723780282655, -3.164135620181696E-8, -0.15087452565065718};
            calculatedGradient = new double[]{0.0, 0.0, 1.27675647831893E-8, 0.0, 0.0, 0.0, 0.0, 1.27675647831893E-8, 0.0, 1.0547118733938987E-8, 0.0, 0.0, 4.440892098500626E-9, 0.0, -0.14291749173001023, 0.0, -0.14593435015974876, -3.164135620181696E-8, -0.15087452565065718};
            numericGradient = smallNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 01!");
            }
            Log.info("passed testSmallGradientNumeric on instance01");

            calculatedGradient = new double[]{0.0, 1.887379141862766E-8, 1.887379141862766E-8, 0.0, 0.0, 0.0, 0.0, 1.887379141862766E-8, 0.0, 1.8318679906315083E-8, 0.0, 0.0, 7.2164496600635175E-9, 0.0, -0.14291749117489871, 0.0, -0.14593435015974876, -4.884981308350689E-8, -0.1477209632394505};
            numericGradient = smallNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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
    public static void testLargeGradientNumericNone() {
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

            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.23221440548226724, 0.0, -0.23534388438051224, 0.0, -0.21879212386277658, 0.0, -0.24007984711360564};
            double[] numericGradient = largeNN.getNumericGradient(instance00);
            Vector.print(numericGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 00!");
            }
            Log.info("passed testLargeGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2322144032618212, 0.0, -0.2353438832702892, 0.0, -0.22038990632466948, 0.0, -0.2400798448931596};
            numericGradient = largeNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 10!");
            }
            Log.info("passed testLargeGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2339974491949448, 0.0, -0.2353438832702892, 0.0, -0.22926425757852087, 0.0, -0.2400798504442747};
            numericGradient = largeNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 01!");
            }
            Log.info("passed testLargeGradientNumeric on instance01");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.885780586188048E-9, 0.0, -0.23399744975005632, 0.0, -0.23534388438051224, 0.0, -0.23045548580569175, -6.106226635438361E-9, -0.24007984711360564};
            numericGradient = largeNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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


    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumericL1() {
        try {
            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(xorData, LossFunction.L1_NORM);

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
            
            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.06250000086915897, 0.0, 0.08749999988455492, 0.0};
            double[] numericGradient = tinyNN.getNumericGradient(instance00);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 00!");
            }
            Log.info("passed testTinyGradientNumeric on instance00");

            calculatedGradient = new double[]{-0.06249522344070613, -0.0872749678082485, 0.0, 0.0, -0.06249522344070613, 0.0, -0.0872749678082485, -0.012488639566932136};
            numericGradient = tinyNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 10!");
            }
            Log.info("passed testTinyGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, -0.06245759021084041, -0.08550240071514281, -0.06245759021084041, 0.0, -0.08550240071514281, -0.03719600183416105};
            numericGradient = tinyNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 01!");
            }
            Log.info("passed testTinyGradientNumeric on instance01");

            calculatedGradient = new double[]{0.06242549310808698, 0.08399108686329981, 0.06242549310808698, 0.08399108686329981, 0.06242549310808698, 0.0, 0.08399108686329981, 0.04928500607626063};
            numericGradient = tinyNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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
    public static void testSmallGradientNumericL1() {
        try {
            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(xorData, LossFunction.L1_NORM);

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

            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14291749173001023, 0.0, 0.15878723891304958, 0.0, 0.1508745262057687};
            double[] numericGradient = smallNN.getNumericGradient(instance00);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 00!");
            }
            Log.info("passed testSmallGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 6.106226635438361E-9, 0.0, 0.0, 0.0, 0.0, 0.0, 6.106226635438361E-9, 0.0, 6.106226635438361E-9, 6.106226635438361E-9, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723835793807, -1.609823385706477E-8, -0.147720963794562};
            numericGradient = smallNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 10!");
            }
            Log.info("passed testSmallGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, 1.27675647831893E-8, 0.0, 0.0, 0.0, 0.0, 1.27675647831893E-8, 0.0, 1.0547118733938987E-8, 0.0, 0.0, 4.440892098500626E-9, 0.0, -0.14291749173001023, 0.0, -0.14593435015974876, -3.164135620181696E-8, -0.15087452565065718};
            numericGradient = smallNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 01!");
            }
            Log.info("passed testSmallGradientNumeric on instance01");

            calculatedGradient = new double[]{0.0, -1.887379141862766E-8, -1.887379141862766E-8, 0.0, 0.0, 0.0, 0.0, -1.887379141862766E-8, 0.0, -1.8318679906315083E-8, 0.0, 0.0, -7.2164496600635175E-9, 0.0, 0.14291749117489871, 0.0, 0.14593435015974876, 4.884981308350689E-8, 0.1477209632394505};
            numericGradient = smallNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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
    public static void testLargeGradientNumericL1() {
        try {
            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(xorData, LossFunction.L1_NORM);

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

            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23221440548226724, 0.0, 0.23534388438051224, 0.0, 0.21879212386277658, 0.0, 0.24007984711360564};
            double[] numericGradient = largeNN.getNumericGradient(instance00);
            Vector.print(numericGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 00!");
            }
            Log.info("passed testLargeGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2322144032618212, 0.0, -0.2353438832702892, 0.0, -0.22038990632466948, 0.0, -0.2400798448931596};
            numericGradient = largeNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 10!");
            }
            Log.info("passed testLargeGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2339974491949448, 0.0, -0.2353438832702892, 0.0, -0.22926425757852087, 0.0, -0.2400798504442747};
            numericGradient = largeNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 01!");
            }
            Log.info("passed testLargeGradientNumeric on instance01");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.885780586188048E-9, 0.0, 0.23399744975005632, 0.0, 0.23534388438051224, 0.0, 0.23045548580569175, 6.106226635438361E-9, 0.24007984711360564};
            numericGradient = largeNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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


    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumericL2() {
        try {
            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(xorData, LossFunction.L2_NORM);

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
            
            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.06250000086915897, 0.0, 0.08749999988455492, 0.0};
            double[] numericGradient = tinyNN.getNumericGradient(instance00);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 00!");
            }
            Log.info("passed testTinyGradientNumeric on instance00");

            calculatedGradient = new double[]{-0.06249522344070613, -0.0872749678082485, 0.0, 0.0, -0.06249522344070613, 0.0, -0.0872749678082485, -0.012488639566932136};
            numericGradient = tinyNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 10!");
            }
            Log.info("passed testTinyGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, -0.06245759021084041, -0.08550240071514281, -0.06245759021084041, 0.0, -0.08550240071514281, -0.03719600183416105};
            numericGradient = tinyNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 01!");
            }
            Log.info("passed testTinyGradientNumeric on instance01");

            calculatedGradient = new double[]{0.06242549310808698, 0.08399108686329981, 0.06242549310808698, 0.08399108686329981, 0.06242549310808698, 0.0, 0.08399108686329981, 0.04928500607626063};
            numericGradient = tinyNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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
    public static void testSmallGradientNumericL2() {
        try {
            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(xorData, LossFunction.L2_NORM);

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

            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14291749173001023, 0.0, 0.15878723891304958, 0.0, 0.1508745262057687};
            double[] numericGradient = smallNN.getNumericGradient(instance00);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 00!");
            }
            Log.info("passed testSmallGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 6.106226635438361E-9, 0.0, 0.0, 0.0, 0.0, 0.0, 6.106226635438361E-9, 0.0, 6.106226635438361E-9, 6.106226635438361E-9, 0.0, 0.0, 0.0, -0.14291749173001023, 0.0, -0.15878723835793807, -1.609823385706477E-8, -0.147720963794562};
            numericGradient = smallNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 10!");
            }
            Log.info("passed testSmallGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, 1.27675647831893E-8, 0.0, 0.0, 0.0, 0.0, 1.27675647831893E-8, 0.0, 1.0547118733938987E-8, 0.0, 0.0, 4.440892098500626E-9, 0.0, -0.14291749173001023, 0.0, -0.14593435015974876, -3.164135620181696E-8, -0.15087452565065718};
            numericGradient = smallNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 01!");
            }
            Log.info("passed testSmallGradientNumeric on instance01");

            calculatedGradient = new double[]{0.0, -1.887379141862766E-8, -1.887379141862766E-8, 0.0, 0.0, 0.0, 0.0, -1.887379141862766E-8, 0.0, -1.8318679906315083E-8, 0.0, 0.0, -7.2164496600635175E-9, 0.0, 0.14291749117489871, 0.0, 0.14593435015974876, 4.884981308350689E-8, 0.1477209632394505};
            numericGradient = smallNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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
    public static void testLargeGradientNumericL2() {
        try {
            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(xorData, LossFunction.L2_NORM);

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

            double[] calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23221440548226724, 0.0, 0.23534388438051224, 0.0, 0.21879212386277658, 0.0, 0.24007984711360564};
            double[] numericGradient = largeNN.getNumericGradient(instance00);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 00!");
            }
            Log.info("passed testLargeGradientNumeric on instance00");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2322144032618212, 0.0, -0.2353438832702892, 0.0, -0.22038990632466948, 0.0, -0.2400798448931596};
            numericGradient = largeNN.getNumericGradient(instance10);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 10!");
            }
            Log.info("passed testLargeGradientNumeric on instance10");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2339974491949448, 0.0, -0.2353438832702892, 0.0, -0.22926425757852087, 0.0, -0.2400798504442747};
            numericGradient = largeNN.getNumericGradient(instance01);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 01!");
            }
            Log.info("passed testLargeGradientNumeric on instance01");

            calculatedGradient = new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.885780586188048E-9, 0.0, 0.23399744975005632, 0.0, 0.23534388438051224, 0.0, 0.23045548580569175, 6.106226635438361E-9, 0.24007984711360564};
            numericGradient = largeNN.getNumericGradient(instance11);
            //Vector.print(numericGradient);
            //Vector.print(calculatedGradient);

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

}

