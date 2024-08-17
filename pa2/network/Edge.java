/**
 * This class represents an Edge in a neural network. It will contain
 * the Edge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import util.Log;

public class Edge {
    //the weight for this edge
    public double weight;

    //the delta calculated by backpropagation for this edge
    public double weightDelta;

    //the input node of this edge
    public RecurrentNode inputNode;

    //the output node of this edge
    public RecurrentNode outputNode;

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingEdge(Edge) and Node.addIncomingEdge(Edge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public Edge(RecurrentNode inputNode, RecurrentNode outputNode) throws NeuralNetworkException {
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        Log.trace("Created a new edge with input " + inputNode.toString() + " and output " + outputNode.toString());

        //initialize the weight and delta to 0
        weight = 0;
        weightDelta = 0;

        inputNode.addOutgoingEdge(this);
        outputNode.addIncomingEdge(this);
    }

    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateBackward(int timeStep, double delta) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        this.weightDelta += delta * inputNode.postActivationValue[timeStep];
        inputNode.delta[timeStep] += delta * this.weight;
    }

    /**
     * Compares if two Edge objects are equal. Two edges are the same
     * if they are from the same input node to the same output node.
     *
     * @return true if the two edge objects have the same input and output nodes
     */
    public boolean equals(Edge other) {
        if (other.inputNode == inputNode && other.outputNode == outputNode) return true;
        return false;
    }

}
