package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import java.util.SortedSet;
import java.util.TreeSet;

import util.Log;

public class CountChars {

    public static HashMap<Character,Integer> charCounts = new HashMap<Character,Integer>();


    public static void main(String[] arguments) {
        try {
            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(arguments[0])));

            String readLine = "";
            //read the file line by line
            while ((readLine = bufferedReader.readLine()) != null) {
                Log.info(readLine); //print out each line of the file if the -DLOG_LEVEL=DEBUG system property is set on the command line

                for (int i = 0; i < readLine.length(); i++) {
                    char c = readLine.charAt(i);
                    
                    if (c == '<') {
                        if (i + 1 < readLine.length()) {
                            if (readLine.charAt(i+1) != 'u') {
                                System.err.println("< did not have 'u' after it!");
                                System.exit(1);
                            }
                        }
                    }

                    if (c == '>') {
                        if (i - 1 >= 0) {
                            if (readLine.charAt(i-1) != 'k') {
                                System.err.println("> did not have 'k' after it!");
                                System.exit(1);
                            }
                        }
                     }

                    if (charCounts.get(c) == null) {
                        charCounts.put(c, 1);
                    } else {
                        charCounts.put(c, charCounts.get(c) + 1);
                    }
                }
            }
            bufferedReader.close();

            int charCount = 0;
            SortedSet<Character> keys = new TreeSet(charCounts.keySet());
            for (Character k : keys) {
                Integer v = charCounts.get(k);
                System.out.println("'" + k + "' - " + v);
                charCount++;
            }
            System.out.println("total number of characters: " + charCount);

        } catch (IOException e) {
            Log.fatal("ERROR converting iris data file");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
