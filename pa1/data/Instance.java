/**
 * This class is used to store one instance of training, validation or testing data.
 */

package data;

public class Instance {
    //neural networks operate with floating point values, so the output
    //and all inputs are doubles
    //these are public so they can be easily accessible by other Classes in our
    //neural network code, however they are also final, so they cannot be modified
    //by other Classes (which could break things).
    public final double[] expectedOutputs;
    public final double[] inputs;

    /**
     * Creates an Instsance object with a given expectedOutputs and number of inputs
     *
     * @param expectedOutputs is the expected output of this instance
     * @param inputs are the input values for this instance
     */
    public Instance(double[] expectedOutputs, double[] inputs) {
        this.expectedOutputs = expectedOutputs;
        this.inputs = inputs;
    }

    /**
     * Compares this Instance to another expected output and inputs to determine
     * if they are the same.
     *
     * @param otherExpectedOutpu another expectedOutputs
     * @param otherInputs another set of inputs
     *
     * @return true if the expected output and inputs provided by the parameters
     * are the same as the ones in this class
     */
    public boolean equals(final double[] otherExpectedOutputs, final double[] otherInputs) {
        //first check to see if the expected outputs have the same length
        //if not then they are not the same and we do not need to check anything
        //else
        if (expectedOutputs.length != otherExpectedOutputs.length) return false;

        for (int i = 0; i < expectedOutputs.length; i++) {
            //if any expected output value is not the same these are not the same
            if (expectedOutputs[i] != otherExpectedOutputs[i]) return false;
        }

        //if the input sizes are different these are not the same and
        //we don't need to compare the individual inputs
        if (inputs.length != otherInputs.length) return false;

        for (int i = 0; i < inputs.length; i++) {
            //if any input value is not the same these are not the same
            if (inputs[i] != otherInputs[i]) return false;
        }

        //everything was the same so these match
        return true;
    }


    /**
     * Compares this Instance to another Instance to determine
     * if they are the same.
     *
     * @param other is another Instance object
     *
     * @return true if the expected output and inputs in this instance
     * and the one passed as a paramter are the same
     */
    public boolean equals(Instance other) {
        return equals(other.expectedOutputs, other.inputs);
    }

    /**
     * This generates a nicely readable string from this Instance
     *
     * @return a String representation of this Instance object
     */
    public String toString() {
        //Using a StringBuilder as they are faster and require less
        //memory than Strings for appending
        StringBuilder sb = new StringBuilder("[");

        for (int i = 0; i < expectedOutputs.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(expectedOutputs[i]);
        }

        sb.append(" : ");

        for (int i = 0; i < inputs.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(inputs[i]);
        }

        sb.append("]");

        return sb.toString();
    }

}
