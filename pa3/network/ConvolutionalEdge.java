/**
 * This class represents an ConvolutionalEdge in a neural network. It will contain
 * the ConvolutionalEdge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import java.util.Random;

import util.Log;

public class ConvolutionalEdge extends Edge {
    //the weight for this edge
    public double weight[][][];

    //the delta calculated by backpropagation for this edge
    public double weightDelta[][][];

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingConvolutionalEdge(ConvolutionalEdge) and Node.addIncomingConvolutionalEdge(ConvolutionalEdge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public ConvolutionalEdge(ConvolutionalNode inputNode, ConvolutionalNode outputNode, int sizeZ, int sizeY, int sizeX) throws NeuralNetworkException {
        super(inputNode, outputNode, sizeZ, sizeY, sizeX);
        this.inputNode = inputNode;
        this.outputNode = outputNode;

        if (inputNode.sizeZ - sizeZ + 1 != outputNode.sizeZ
                || inputNode.sizeY - sizeY + 1 != outputNode.sizeY - (2 * outputNode.padding)
                || inputNode.sizeX - sizeX + 1 != outputNode.sizeX - (2 * outputNode.padding)) {
            throw new NeuralNetworkException("Cannot connect input node " + inputNode.toString() + " to output node " + outputNode.toString() + " because sizes do not work with this filter (" + sizeZ + "x" + sizeY + "x" + sizeX  + "), output node size should be (batchSize x" + (inputNode.sizeZ - sizeZ + 1) + "x" + (inputNode.sizeY - sizeY + 1) + "x" + (inputNode.sizeX - sizeX + 1) + ")");
        }

        //initialize the weight and delta to 0
        weight = new double[sizeZ][sizeY][sizeX];
        weightDelta = new double[sizeZ][sizeY][sizeX];
    }

    /**
     * Resets the deltas for this edge
     */
    public void reset() {
        //Log.info("resetting convolutional edge with sizeZ: " + sizeZ + ", sizeY: " + sizeY + ", sizeX: " + sizeX);
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weightDelta[z][y][x] = 0;
                }
            }
        }
    }

    /**
     * Used to get the weights of this Edge.
     * It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weights[position + weightCount] = weight[z][y][x];
                    weightCount++;
                }
            }
        }

        return weightCount;
    }

    /**
     * Used to print gradients related to this edge, along with informationa
     * about this edge.
     * It start printing the gradients passed in starting at position, and 
     * return the number of gradients it printed.
     *
     * @param position is the index to start printing different gradients
     * @param numericGradient is the array of the numeric gradient we're printing
     * @param backpropGradient is the array of the backprop gradient we're printing
     *
     * @return the number of gradients printed by this edge
     */
    public int printGradients(int position, double[] numericGradient, double[] backpropGradient) {
        //don't print anything out, but print out this edge
        Log.info("ConvolutionalEdge from Node [layer: " + inputNode.layer + ", number: " + inputNode.number + "] to Node [layer: " + outputNode.layer + ", number: " + outputNode.number + "] to Node:");

        int count = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    Log.info("\tweights[" + z + "][" + y + "][" + x + "]: "+ Log.twoGradients(numericGradient[position + count], backpropGradient[position + count]));
                    count++;
                }
            }
        }

        return count;
    }


    /**
     * Used to get the deltas of this Edge.
     * It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    deltas[position + deltaCount] = weightDelta[z][y][x];
                    deltaCount++;
                }
            }
        }

        return deltaCount;
    }


    /**
     * Used to set the weights of this Edge.
     * It uses the same technique as Node.getWeights
     * where the starting position of weights to set is passed, and it returns
     * how many weights were set.
     * 
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */
    public int setWeights(int position, double[] weights) {
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weight[z][y][x] = weights[position + weightCount];
                    weightCount++;
                }
            }
        }

        return weightCount;
    }



    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * the range calculated by it's output node (which should be sqrt(2)/sqrt(all incoming edge filter sizes).
     *
     * @param range is sqrt(2)/sqrt(sum of output node incoming filter sizes)
     */
    @Override
    public void initializeKaiming(double range, int fanIn) {
        // TODO: Implement this for Programming Assignment 3 - Part 1
        Random random = new Random();
        for(int ch = 0; ch < this.weight.length; ch++) {
            for(int col = 0; col < this.weight[ch].length; col++) {
                for(int row = 0; row < this.weight[ch][col].length; row++){
                    this.weight[ch][col][row] = range * random.nextGaussian();
                }
            }
        }
    }

    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * uniformly within the range calculated by it's output node (which 
     * should be between negative and positive sqrt(6)/sqrt(all incoming 
     * and outgoing edge filter sizes).
     *
     * @param range is sqrt(6)/sqrt(sum of output node incoming and outgoing filter sizes)
     */
    @Override
    public void initializeXavier(double range, int fanIn, int fanOut) {
        // TODO: Implement this for Programming Assignment 3 - Part 1
        Random random = new Random();
        for(int ch = 0; ch < this.weight.length; ch++) {
            for(int col = 0; col < this.weight[ch].length; col++) {
                for(int row = 0; row < this.weight[ch][col].length; row++){
                    this.weight[ch][col][row] = range * (2.0 * random.nextGaussian() - 1);
                }
            }
        }

    }


    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateForward(double[][][][] inputValues) {
        // TODO: You need to implement this for Programming Assignment 3 - Part 1
        for (int img = 0; img < this.outputNode.inputValues.length; img++) {
            for (int ch = 0; ch < this.outputNode.inputValues[img].length; ch++) {
                for (int col = 0; col < this.outputNode.inputValues[img][ch].length - 2 * this.outputNode.padding; col++) {
                    for (int row = 0; row < this.outputNode.inputValues[img][ch][col].length - 2 * this.outputNode.padding; row++) {
                        for (int fCh = 0; fCh < this.weight.length; fCh++) {
                            for (int fc = 0; fc < this.weight[fCh].length; fc++) {
                                for (int fr = 0; fr < this.weight[fCh][fc].length; fr++) {
                                    this.outputNode.inputValues[img][ch][col + this.outputNode.padding][row + this.outputNode.padding] += this.weight[fCh][fc][fr] * inputValues[img][ch + fCh][col + fc][row + fr];
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateBackward(double[][][][] delta) {
        // TODO: You need to implement this for Programming Assignment 3 - Part 2
        for (int img = 0; img < delta.length; img++) {
            for (int ch = 0; ch < delta[img].length; ch++) {
                for (int col = 0; col < delta[img][ch].length - 2 * this.outputNode.padding; col++) {
                    for (int row = 0; row < delta[img][ch][col].length - 2 * this.outputNode.padding; row++) {
                        double pDelta = delta[img][ch][col + this.outputNode.padding][row + this.outputNode.padding];
                        for (int fCh = 0; fCh < weightDelta.length; fCh++) {
                            for (int fc = 0; fc < this.weightDelta[fCh].length; fc++) {
                                for (int fr = 0; fr < this.weightDelta[fCh][fc].length; fr++) {
                                    this.weightDelta[fCh][fc][fr] += pDelta * inputNode.outputValues[img][ch + fCh][col + fc][row + fr];
                                    this.inputNode.delta[img][ch][col + fc][row + fr] += pDelta * this.weight[fCh][fc][fr];

                                }
                            }
                        }

                    }
                }
            }
        }

    }

}
