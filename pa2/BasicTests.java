/**
 * A helpful class of tests to make sure programming assignment 1
 * part 1 is working correctly
 */

import java.util.Arrays;
import java.util.List;

import data.Sequence;
import data.SequenceDataSet;
import data.CharacterSequence;
import data.SequenceException;

import network.LossFunction;
import network.RecurrentNeuralNetwork;
import network.RNNNodeType;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;

public class BasicTests {

    public static void main(String[] arguments) {
        testSequenceDataSet(new SequenceDataSet("penntreebank train small", "./datasets/penntreebank_train_small.txt"));
        testSequenceDataSet(new SequenceDataSet("penntreebank train full", "./datasets/penntreebank_train_full.txt"));

        testSequenceDataSet(new SequenceDataSet("penntreebank test small", "./datasets/penntreebank_test_small.txt"));
        testSequenceDataSet(new SequenceDataSet("penntreebank test full", "./datasets/penntreebank_test_full.txt"));

        testSequenceDataSet(new SequenceDataSet("penntreebank valid small", "./datasets/penntreebank_valid_small.txt"));
        testSequenceDataSet(new SequenceDataSet("penntreebank valid full", "./datasets/penntreebank_valid_full.txt"));

    }

    /**
     * Tests that generating the CharacterSequence and SequenceDataSet objects
     * is working correctly, in particular converting between the character values
     * and their int representations needed for the neural networks
     *
     * @param dataSet is the SequenceDataSet to test
     */
    public static void testSequenceDataSet(SequenceDataSet dataSet) {
        List<Sequence> sequences = dataSet.getSequences();

        boolean hadErrors = false;
        for (int i = 0; i < sequences.size(); i++) {
            CharacterSequence sequence = (CharacterSequence)sequences.get(i);

            try {
                String sequenceStr = sequence.getSequence();
                Log.trace("testing encode/decode for: '" + sequenceStr + "'");

                int[] encoding = CharacterSequence.encode(sequenceStr);
                String decodedSequence = CharacterSequence.decode(encoding);

                Log.trace("encoding is: " + Arrays.toString(encoding));

                if (!sequenceStr.equals(decodedSequence)) {
                    Log.error("Encoding/decoding of character sequences had a problem for " + dataSet.getName());
                    Log.error("CharacterSequence before encode/decode: '" + sequenceStr + "'");
                    Log.error("CharacterSequence after  encode/decode: '" + decodedSequence + "'");
                    hadErrors = true;
                }
            } catch (SequenceException e) {
                Log.error("Encoding/decoding of character sequences had a problem for " + dataSet.getName());
                Log.error("Threw exception: " + e);
                e.printStackTrace();
                hadErrors = true;
            }
        }

        if (hadErrors) {
            Log.error("FAILED testSequenceDataSet on: " + dataSet.getName());
        } else {
            Log.info("PASSED testSequenceDataSet on: " + dataSet.getName());
        }
    }

    /**
     * Checks to make sure that the RecurrentNeuralNetwork.getWeights() and 
     * RecurrentNeuralNetwork.setWeights(double[]) methods work correctly. After the 
     * weights are set with setWeights they should be returned identically from getWeights.
     *
     * @param network is the neural network to test
     * @param networkName is a friendly user readable name for the network
     *
     * @throws NeuralNetworkException if the weights were not the same or if there was a problem
     * with getWeights or setWeights
     */
    public static void checkGetSetWeights(RecurrentNeuralNetwork network, String networkName) throws NeuralNetworkException {
        Log.debug("Testing get/set weights on neural network '" + networkName + "'");
        int numberWeights = network.getNumberWeights();
        double[] testWeights = new double[numberWeights];
        for (int i = 0; i < numberWeights; i++) {
            testWeights[i] = i;
        }

        network.setWeights(testWeights);

        double[] testWeights2 = network.getWeights();

        if (testWeights.length != testWeights2.length) {
            throw new NeuralNetworkException("Failed getSetWeights test on " + networkName 
                    + ", testWeights.length was " + testWeights.length + " and testWeights2.length was " + testWeights2.length);
        }

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
    public static boolean closeEnough(double[] v1, double[] v2) {
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
    public static boolean gradientsCloseEnough(double[] g1, double[] g2, String[] weightNames) {
        if (Vector.norm(g1) == 0.0 && Vector.norm(g2) == 0.0) return true;

        double relativeError = Vector.norm(Vector.subtractVector(g1, g2)) / Math.max(Vector.norm(g1), Vector.norm(g2));

        if (relativeError >= 1e-4) {
            Log.error("relativeError bad: " + relativeError);
            for (int i = 0; i < g1.length; i++) {
                Log.error("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]) + ", " + weightNames[i]);
            }

        } else if (relativeError >= 1e-5) {
            Log.warning("relativeError probably bad: " + relativeError);
            for (int i = 0; i < g1.length; i++) {
                Log.trace("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]) + ", " + weightNames[i]);
            }

        } else if (relativeError >= 1e-7) {
            Log.debug("relativeError might be bad: " + relativeError);
            for (int i = 0; i < g1.length; i++) {
                Log.trace("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]) + ", " + weightNames[i]);
            }
        }

        Log.info("relativeError: " + relativeError);
        return relativeError <= 1e-4;
    }

}
