/**
 * This class represents an RecurrentEdge in a neural network. It will contain
 * the RecurrentEdge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import util.Log;

public class RecurrentEdge {
    //the weight for this edge
    public double weight;

    //the delta calculated by backpropagation for this edge
    public double weightDelta;

    //the input node of this edge
    public RecurrentNode inputNode;

    //the output node of this edge
    public RecurrentNode outputNode;

    public int timeSkip;

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingRecurrentEdge(RecurrentEdge) and Node.addIncomingRecurrentEdge(RecurrentEdge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public RecurrentEdge(RecurrentNode inputNode, RecurrentNode outputNode, int timeSkip) throws NeuralNetworkException {
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        this.timeSkip = timeSkip;
        Log.trace("Created a new recurrent edge with input " + inputNode.toString() + " and output " + outputNode.toString() + " and time skip " +  timeSkip);

        //initialize the weight and delta to 0
        weight = 0;
        weightDelta = 0;

        inputNode.addOutgoingRecurrentEdge(this);
        outputNode.addIncomingRecurrentEdge(this);
    }

    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateBackward(int timeStep, double delta) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        //HINT: don't propagate backward to times that don't exist
        if(timeStep - timeSkip >= 0){
            this.weightDelta += delta * inputNode.postActivationValue[timeStep - timeSkip];
            inputNode.delta[timeStep - timeSkip] += delta * weight;
        }
    }

    /**
     * Compares if two RecurrentEdge objects are equal. Two recurrent edges are the same
     * if they are from the same input node to the same output node and their time skip is
     * the same.
     *
     * @return true if the two recurrent edge objects have the same input and output nodes and the same time skip
     */
    public boolean equals(RecurrentEdge other) {
        if (other.inputNode == inputNode && other.outputNode == outputNode && other.timeSkip == timeSkip) return true;
        return false;
    }


}
