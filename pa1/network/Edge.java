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
    public Node inputNode;

    //the output node of this edge
    public Node outputNode;

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingEdge(Edge) and Node.addIncomingEdge(Edge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public Edge(Node inputNode, Node outputNode) throws NeuralNetworkException {
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        Log.debug("Created a new edge with input " + inputNode.toString() + " and output " + outputNode.toString());

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
    public void propagateBackward(double delta) {
        //Uncommenting the following may help you debug this method:
        //System.out.println("Edge with output node[layer " + outputNode.layer + ", number " + outputNode.number + "] and input node[layer " + inputNode.layer + ", number " + inputNode.number +"] backpropagating delta: " + delta);

        //TODO: You need to implement this for Programming Assignment 1 - Part 2
        //Calculate the delta for the input node and update the weightDelta for this edge.
        weightDelta += (delta * inputNode.postActivationValue);
        inputNode.delta += (delta * this.weight);
    }

    /**
     * Checks to see if two edges are equal in that they have the same input and
     * output node.
     *
     * @param other is the other Edge object to compare against
     *
     * @return true if the edges have the same input and output nodes
     */
    public boolean equals(Edge other) {
        return this.inputNode.layer == other.inputNode.layer && this.inputNode.number == other.inputNode.number
                && this.outputNode.layer == other.outputNode.layer && this.outputNode.number == other.outputNode.number;
    }


}
