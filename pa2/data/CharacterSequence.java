package data;


import util.Log;

public class CharacterSequence implements Sequence {

    /**
     * The string for this character sequence
     */
    String sequence;

    /**
     * An integer encoding of the string sequence.
     * The different possible characters are (we will count <unk> 
     * as a single character):
     *
     * 0-9      -- numbers, encoded value will be 0 - 9
     * a-z      -- lower case characters, encoded values will be 10 - 36
     * ' '      -- space, encoded value will be 37
     * '#'      -- number sign, encoded value will be 38
     * '$'      -- dollar sign, encoded value will be 39
     * '&'      -- ampersand, encoded value will be 40
     * '''      -- single quote, encoded value will be 41
     * '*'      -- star, encoded value will be 42
     * '-'      -- dash, encoded value will be 43
     * '.'      -- period, encoded value will be 44
     * '/'      -- forward slash, encoded value will be 45
     * '\'      -- backward slash, encoded value will be 46
     * N        -- N (for number), encoded value will be 47
     * <unk>    -- rare words like names have been replaced with <unk>, we will use ? as a standin for <unk>, encoded value will be 48
     * 
     * for a total of 48 characters
     */
    int[] encoding;

    /**
     * Encodes a character sequence into a number of ints, and replaces <unk> with '?'
     *
     * @param line a line from the character training file
     *
     */
    public CharacterSequence(String line) throws SequenceException {
        Log.trace("line: '" + line + "'");
        sequence = line.replaceAll("<unk>", "?");
        Log.trace("line with <unk> replaced by ?: '" + sequence + "'");

        encoding = encode(sequence);
    }

    /**
     * Returns the encoded int value at the given time step.
     *
     * @param timeStep the timeStep of the int value to return
     *
     * @return the encoded int value of the sequence at the given time step
     */
    public int valueAt(int timeStep) {
        return encoding[timeStep];
    }

    /**
     * Returns the character value at the given time step.
     *
     * @param timeStep the timeStep of the int value to return
     *
     * @return the character of the sequence at the given time step
     */
    public int charAt(int timeStep) {
        return sequence.charAt(timeStep);
    }


    /**
     * @return the length of this sequence
     */
    public int getLength() {
        return sequence.length();
    }

    /**
     * @return the String sequence for this CharacterSequence
     */
    public String getSequence() {
        return sequence;
    }

    /**
     * @return the int array encoding for this CharacterSequence
     */
    public int[] getEncoding() {
        return encoding;
    }

    /**
     * Convert the sequence of chars to ints that we can use to train a neural network
     *
     * @param sequence is the String to convert
     *
     * @return an array of ints, each character converted to the appropriate int
     */
    public static int[] encode(String sequence) throws SequenceException {
        int[] encoding = new int[sequence.length()];

        for (int i = 0; i < sequence.length(); i++) {
            encoding[i] = charToInt(sequence.charAt(i));
        }

        return encoding;
    }

    /**
     * Convert an array of ints to the String that should have been used to
     * generate them.
     */
    public static String decode(int[] encoding) throws SequenceException {
        //use a string builder to quickly append all the chars to a single
        //object
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < encoding.length; i++) {
            sb.append(intToChar(encoding[i]));
        }

        return sb.toString();
    }


    /**
     * Converts a sequence character into an int representation
     *
     * @param c the character to convert
     *
     * @return an int for that character (see above conversion)
     * @throws SequenceException on an unknown character
     */
    public static int charToInt(char c) throws SequenceException {
        int asciiValue = (int)c;

        switch (c) {
            case '0': return 0;
            case '1': return 1;
            case '2': return 2;
            case '3': return 3;
            case '4': return 4;
            case '5': return 5;
            case '6': return 6;
            case '7': return 7;
            case '8': return 8;
            case '9': return 9;
            case 'a': return 10;
            case 'b': return 11;
            case 'c': return 12;
            case 'd': return 13;
            case 'e': return 14;
            case 'f': return 15;
            case 'g': return 16;
            case 'h': return 17;
            case 'i': return 18;
            case 'j': return 19;
            case 'k': return 20;
            case 'l': return 21;
            case 'm': return 22;
            case 'n': return 23;
            case 'o': return 24;
            case 'p': return 25;
            case 'q': return 26;
            case 'r': return 27;
            case 's': return 28;
            case 't': return 29;
            case 'u': return 30;
            case 'v': return 31;
            case 'w': return 32;
            case 'x': return 33;
            case 'y': return 34;
            case 'z': return 35;
            case ' ': return 36;
            case '#': return 37;
            case '$': return 38;
            case '&': return 39;
            case '\'': return 40;
            case '*': return 41;
            case '-': return 42;
            case '.': return 43;
            case '/': return 44;
            case '\\': return 45;
            case 'N': return 46;
            case '?': return 47;
            default:
                      throw new SequenceException("Could not convert an unknown character '" + c + "' to an int while generating a CharacterSequence!");
        }
    }

    /**
     * Converts an an int into a sequence character 
     *
     * @param v the int value to convert
     *
     * @return an character for that int (see above conversion)
     * @throws SequenceException on an unknown int  
     */
    public static char intToChar(int v) throws SequenceException {
        switch (v) {
            case 0: return '0';
            case 1: return '1';
            case 2: return '2';
            case 3: return '3';
            case 4: return '4';
            case 5: return '5';
            case 6: return '6';
            case 7: return '7';
            case 8: return '8';
            case 9: return '9';
            case 10: return 'a';
            case 11: return 'b';
            case 12: return 'c';
            case 13: return 'd';
            case 14: return 'e';
            case 15: return 'f';
            case 16: return 'g';
            case 17: return 'h';
            case 18: return 'i';
            case 19: return 'j';
            case 20: return 'k';
            case 21: return 'l';
            case 22: return 'm';
            case 23: return 'n';
            case 24: return 'o';
            case 25: return 'p';
            case 26: return 'q';
            case 27: return 'r';
            case 28: return 's';
            case 29: return 't';
            case 30: return 'u';
            case 31: return 'v';
            case 32: return 'w';
            case 33: return 'x';
            case 34: return 'y';
            case 35: return 'z';
            case 36: return ' ';
            case 37: return '#';
            case 38: return '$';
            case 39: return '&';
            case 40: return '\'';
            case 41: return '*';
            case 42: return '-';
            case 43: return '.';
            case 44: return '/';
            case 45: return '\\';
            case 46: return 'N';
            case 47: return '?';
            default:
                     throw new SequenceException("Could not convert an unknown int '" + v + "' to a character as part of a CharacterSequence!");
         }
    }
}
