package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import util.Log;

public class ConvertIris {
    public static void main(String[] arguments) {
        try {
            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File("./datasets/iris.data")));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("./datasets/iris.txt")));

            String readLine = "";
            //read the file line by line
            while ((readLine = bufferedReader.readLine()) != null) {
                Log.info(readLine); //print out each line of the file if the -DLOG_LEVEL=DEBUG system property is set on the command line

                if (readLine.length() == 0 || readLine.charAt(0) == '#') {
                    //empty lines are skipped, as well as lines beginning 
                    //with the '#' character, these are comments and also skipped
                    continue;
                }

                String[] values = readLine.split(",");
                String sampleClass = values[values.length - 1]; //the class of the dataset is the last column

                //put everything in a stringbuffer before writing to the file
                StringBuffer sb = new StringBuffer();

                //This dataset has three classes: 'Iris-setosa', 'Iris-versicolor', and 'Iris-virginica'
                if (sampleClass.equals("Iris-setosa")) {
                    sb.append("0"); //this will be the third class
                } else if (sampleClass.equals("Iris-versicolor")) {
                    sb.append("1"); //this will be the second class
                } else if (sampleClass.equals("Iris-virginica")) {
                    sb.append("2"); //this will be the third class
                } else {
                    System.err.println("ERROR: unknown class in iris.data file: '" + sampleClass + "'");
                    System.err.println("This should not happen.");
                    System.exit(1);
                }
                sb.append(":");

                //the other values are the different input values for the neural network
                for (int i = 0; i < values.length - 1; i++) {
                    if (i > 0) sb.append(",");
                    sb.append(values[i]);
                }
                sb.append("\n");

                Log.info(sb.toString());
                bufferedWriter.write(sb.toString());
            }
            bufferedWriter.close();
            bufferedReader.close();

        } catch (IOException e) {
            Log.fatal("ERROR converting iris data file");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
