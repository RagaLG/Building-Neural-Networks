package data;

import java.io.IOException;
import java.util.Arrays;
import util.Log;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

public class ConvertMushroom {
    public static void main(String[] arguments) {
        try {
            //create a buffered reader given the filename
            Log.info("Opened reader for file agaricus-lepiota.data.");
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File("./datasets/agaricus-lepiota.data")));
            Log.info("Opened writer for file agaricus-lepiota-ohe.txt.");
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("./datasets/agaricus-lepiota-ohe.txt")));

            String readLine = "";
            String put = "";
            //reading file
            int numLines = 0;
            while ((readLine = bufferedReader.readLine()) != null) {
                Log.debug("Processing line: " + readLine);
                String[] line = readLine.split(",");
                Log.debug("Split string array from line: "+Arrays.toString(line));
                put= "";
                Log.debug("Processing column "+0+": "+line[0]);
                if (line[0].equals("p"))
                    put += "1";
                else if (line[0].equals("e")) {
                    put += "0";
                } else {
                    System.out.println(line[0]);
                    System.out.println("invalid");
                    System.exit(1);
                }
                put += ":";
                for (int i = 1; i < line.length; i++) {
                    Log.debug("Processing column: " + i + ": "+line[i]);
                    String encodedOut = oneHot(i,line[i]);
                    if (i < line.length - 1 && !encodedOut.equals("invalid"))
                        put += encodedOut + ",";
                    else if(i == line.length-1 && !encodedOut.equals("invalid")) {
                        put += encodedOut+"\n";
                        bufferedWriter.write(put);
                    }
                    else
                        Log.error(""+i+" "+line[i]+"   "+encodedOut);
                }
                numLines++;
            }
            bufferedWriter.close();
            bufferedReader.close();
            Log.info("Closed reader and writer.");
            Log.info("Processed "+numLines+" lines.");
        } catch (IOException e) {
            Log.fatal("ERROR converting agaricus-lepiota data file");
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static String oneHot(int posData,String columnValue) {
        switch (posData) {
            case 1: {
                switch (columnValue) {
                    case "b":    return "1,0,0,0,0,0";
                    case "c":    return "0,1,0,0,0,0";
                    case "x":    return "0,0,1,0,0,0";
                    case "f":    return "0,0,0,1,0,0";
                    case "k":    return "0,0,0,0,1,0";
                    case "s":    return "0,0,0,0,0,1";
                }
            }
            case 2:{
                switch (columnValue){
                    case "f":    return "1,0,0,0";
                    case "g":    return "0,1,0,0";
                    case "y":    return "0,0,1,0";
                    case "s":    return "0,0,0,1";
                }
            }
            case 3:{
                switch (columnValue){
                    case "n":     return "1,0,0,0,0,0,0,0,0,0";
                    case "b":     return "0,1,0,0,0,0,0,0,0,0";
                    case "c":     return "0,0,1,0,0,0,0,0,0,0";
                    case "g":     return "0,0,0,1,0,0,0,0,0,0";
                    case "r":     return "0,0,0,0,1,0,0,0,0,0";
                    case "p":     return "0,0,0,0,0,1,0,0,0,0";
                    case "u":     return "0,0,0,0,0,0,1,0,0,0";
                    case "e":     return "0,0,0,0,0,0,0,1,0,0";
                    case "w":     return "0,0,0,0,0,0,0,0,1,0";
                    case "y":     return "0,0,0,0,0,0,0,0,0,1";
                }
            }
            case 4: {
                switch (columnValue){
                    case "t":     return "1,0";
                    case "f":     return "0,1";
                }
            }
            case 5: {
                switch (columnValue){
                    case "a":     return "1,0,0,0,0,0,0,0,0";
                    case "l":     return "0,1,0,0,0,0,0,0,0";
                    case "c":     return "0,0,1,0,0,0,0,0,0";
                    case "y":     return "0,0,0,1,0,0,0,0,0";
                    case "f":     return "0,0,0,0,1,0,0,0,0";
                    case "m":     return "0,0,0,0,0,1,0,0,0";
                    case "n":     return "0,0,0,0,0,0,1,0,0";
                    case "p":     return "0,0,0,0,0,0,0,1,0";
                    case "s":     return "0,0,0,0,0,0,0,0,1";
                }
            }
            case 6: {
                switch (columnValue){
                    case "a":     return "1,0,0,0";
                    case "d":     return "0,1,0,0";
                    case "f":     return "0,0,1,0";
                    case "n":     return "0,0,0,1";
                }
            }
            case 7: {
                switch (columnValue){
                    case "c":     return "1,0,0";
                    case "w":     return "0,1,0";
                    case "d":     return "0,0,1";
                }
            }
            case 8:{
                switch (columnValue){
                    case "b":     return "1,0";
                    case "n":     return "0,1";
                }
            }

            case 9:{
                switch (columnValue){
                    case "k":     return "1,0,0,0,0,0,0,0,0,0,0,0";
                    case "n":     return "0,1,0,0,0,0,0,0,0,0,0,0";
                    case "b":     return "0,0,1,0,0,0,0,0,0,0,0,0";
                    case "h":     return "0,0,0,1,0,0,0,0,0,0,0,0";
                    case "g":     return "0,0,0,0,1,0,0,0,0,0,0,0";
                    case "r":     return "0,0,0,0,0,1,0,0,0,0,0,0";
                    case "o":     return "0,0,0,0,0,0,1,0,0,0,0,0";
                    case "p":     return "0,0,0,0,0,0,0,1,0,0,0,0";
                    case "u":     return "0,0,0,0,0,0,0,0,1,0,0,0";
                    case "e":     return "0,0,0,0,0,0,0,0,0,1,0,0";
                    case "w":     return "0,0,0,0,0,0,0,0,0,0,1,0";
                    case "y":     return "0,0,0,0,0,0,0,0,0,0,0,1";
                }
            }

            case 10:{
                switch (columnValue){
                    case "e":     return "1,0";
                    case "t":     return "0,1";
                }
            }
            case 11:{
                switch (columnValue){
                    case "b":     return "1,0,0,0,0,0,0";
                    case "c":     return "0,1,0,0,0,0,0";
                    case "u":     return "0,0,1,0,0,0,0";
                    case "e":     return "0,0,0,1,0,0,0";
                    case "z":     return "0,0,0,0,1,0,0";
                    case "r":     return "0,0,0,0,0,1,0";
                    case "?":     return "0,0,0,0,0,0,1";
                }
            }
            case 12:{
                switch (columnValue){
                    case "f":     return "1,0,0,0";
                    case "y":     return "0,1,0,0";
                    case "k":     return "0,0,1,0";
                    case "s":     return "0,0,0,1";
                }
            }
            case 13:{
                switch (columnValue){
                    case "f":     return "1,0,0,0";
                    case "y":     return "0,1,0,0";
                    case "k":     return "0,0,1,0";
                    case "s":     return "0,0,0,1";
                }
            }
            case 14:{
                switch (columnValue){
                    case "n":     return "1,0,0,0,0,0,0,0,0";
                    case "b":     return "0,1,0,0,0,0,0,0,0";
                    case "c":     return "0,0,1,0,0,0,0,0,0";
                    case "g":     return "0,0,0,1,0,0,0,0,0";
                    case "o":     return "0,0,0,0,1,0,0,0,0";
                    case "p":     return "0,0,0,0,0,1,0,0,0";
                    case "e":     return "0,0,0,0,0,0,1,0,0";
                    case "w":     return "0,0,0,0,0,0,0,1,0";
                    case "y":     return "0,0,0,0,0,0,0,0,1";
                }
            }
            case 15:{
                switch (columnValue){
                    case "n":     return "1,0,0,0,0,0,0,0,0";
                    case "b":     return "0,1,0,0,0,0,0,0,0";
                    case "c":     return "0,0,1,0,0,0,0,0,0";
                    case "g":     return "0,0,0,1,0,0,0,0,0";
                    case "o":     return "0,0,0,0,1,0,0,0,0";
                    case "p":     return "0,0,0,0,0,1,0,0,0";
                    case "e":     return "0,0,0,0,0,0,1,0,0";
                    case "w":     return "0,0,0,0,0,0,0,1,0";
                    case "y":     return "0,0,0,0,0,0,0,0,1";
                }
            }
            case 16:{
                switch (columnValue){
                    case "p":     return "1,0";
                    case "u":     return "0,1";
                }
            }
            case 17:{
                switch (columnValue){
                    case "n":     return "1,0,0,0";
                    case "o":     return "0,1,0,0";
                    case "w":     return "0,0,1,0";
                    case "y":     return "0,0,0,1";
                }
            }
            case 18:{
                switch (columnValue){
                    case "n":     return "1,0,0";
                    case "o":     return "0,1,0";
                    case "t":     return "0,0,1";
                }
            }
            case 19:{
                switch (columnValue){
                    case "c":     return "1,0,0,0,0,0,0,0";
                    case "e":     return "0,1,0,0,0,0,0,0";
                    case "f":     return "0,0,1,0,0,0,0,0";
                    case "l":     return "0,0,0,1,0,0,0,0";
                    case "n":     return "0,0,0,0,1,0,0,0";
                    case "p":     return "0,0,0,0,0,1,0,0";
                    case "s":     return "0,0,0,0,0,0,1,0";
                    case "z":     return "0,0,0,0,0,0,0,1";
                }
            }
            case 20:{
                switch (columnValue){
                    case "k":     return "1,0,0,0,0,0,0,0,0";
                    case "n":     return "0,1,0,0,0,0,0,0,0";
                    case "b":     return "0,0,1,0,0,0,0,0,0";
                    case "h":     return "0,0,0,1,0,0,0,0,0";
                    case "r":     return "0,0,0,0,1,0,0,0,0";
                    case "o":     return "0,0,0,0,0,1,0,0,0";
                    case "u":     return "0,0,0,0,0,0,1,0,0";
                    case "w":     return "0,0,0,0,0,0,0,1,0";
                    case "y":     return "0,0,0,0,0,0,0,0,1";
                }
            }
            case 21:{
                switch (columnValue){
                    case "a":     return "1,0,0,0,0,0";
                    case "c":     return "0,1,0,0,0,0";
                    case "n":     return "0,0,1,0,0,0";
                    case "s":     return "0,0,0,1,0,0";
                    case "v":     return "0,0,0,0,1,0";
                    case "y":     return "0,0,0,0,0,1";
                }
            }
            case 22:{
                switch (columnValue){
                    case "g":     return "1,0,0,0,0,0,0";
                    case "l":     return "0,1,0,0,0,0,0";
                    case "m":     return "0,0,1,0,0,0,0";
                    case "p":     return "0,0,0,1,0,0,0";
                    case "u":     return "0,0,0,0,1,0,0";
                    case "w":     return "0,0,0,0,0,1,0";
                    case "d":     return "0,0,0,0,0,0,1";
                }
            }
        }
        return "invalid";
    }
}
