package data;

import java.util.List;

public interface DataSet {

    //returns the number of input values for the data set 
    int getNumberInputs();

    //returns the number of output values for the data set 
    int getNumberOutputs();

    //returns the maximum length of a sequence
    int getMaxLength();

    //shuffles the instances in the data set
    void shuffle();

    //gets the number of sequences/time series in the data set
    int getNumberSequences();

    Sequence getSequence(int position);
    List<Sequence> getSequences(int position, int numberOfSequences);
    List<Sequence> getSequences();

}
