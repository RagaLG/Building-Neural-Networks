/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class PA14Tests {
    public static DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

    public static void main(String[] arguments) {
        if (arguments.length != 1) {
            System.err.println("Invalid arguments, you must specify a loss function, usage: ");
            System.err.println("\tjava PA14Tests <loss function>");
            System.err.println("\tloss function options are: 'svm' or 'softmax'");
            System.exit(1);
        }

        testNormalize();

        String lossFunctionName = arguments[0];

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("svm")) {
            Log.info("Using an SVM loss function.");
            lossFunction = LossFunction.SVM;

            //test the numeric gradient calculations on tiny,
            //small and large neural networks with L2_NORM
            //loss functions
            testTinyGradientNumericSVM();
            testSmallGradientNumericSVM();
            testLargeGradientNumericSVM();

        } else if (lossFunctionName.equals("softmax")) {
            Log.info("Using an SOFTMAX loss function.");
            lossFunction = LossFunction.SOFTMAX;

            //test the numeric gradient calculations on tiny,
            //small and large neural networks with L2_NORM
            //loss functions
            testTinyGradientNumericSOFTMAX();
            testSmallGradientNumericSOFTMAX();
            testLargeGradientNumericSOFTMAX();

        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        //these tests calculation of of the gradient via
        //the backwards pass for the tiny, small and large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights when the network has a L2_NORM
        //loss function
        PA12Tests.testTinyGradients(irisData, lossFunction);
        PA12Tests.testSmallGradients(irisData, lossFunction);
        PA12Tests.testLargeGradients(irisData, lossFunction);

        //this tests calculation of of the gradient via
        //the backwards pass for the tiny, small and large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights when the network has a L2_NORM
        //loss function
        PA12Tests.testTinyGradientsMultiInstance(irisData, lossFunction);
        PA12Tests.testSmallGradientsMultiInstance(irisData, lossFunction);
        PA12Tests.testLargeGradientsMultiInstance(irisData, lossFunction);
     }

    public static void testNormalize() {
        double[] means = irisData.getInputMeans();
        double[] stdDevs = irisData.getInputStandardDeviations();
        Log.info("calculated data set means: " + Arrays.toString(means));
        Log.info("calculated data set standard deviations: " + Arrays.toString(stdDevs));
        irisData.normalize(means, stdDevs);

        double[] actualMeans = new double[]{5.843333333333335, 3.0540000000000007, 3.7586666666666693, 1.1986666666666672};
        double[] actualStdDevs = new double[]{0.8280661279778629, 0.4335943113621737, 1.7644204199522617, 0.7631607417008414};

        if (BasicTests.vectorsCloseEnough(means, actualMeans)) {
            Log.info("passed mean calculation from iris data set!");
        } else {
            Log.info("failed calculation from iris data set!");
        }

        if (BasicTests.vectorsCloseEnough(stdDevs, actualStdDevs)) {
            Log.info("passed standard deviation calculation from iris data set!");
        } else {
            Log.info("failed standard deviation calculation from iris data set!");
        }
     }

    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumericSVM() {
        try {
            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(irisData, LossFunction.SVM);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[tinyNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            tinyNN.setWeights(weights);
            
            double[] calculatedGradient = new double[]{-0.04651449669879071, -0.06285393094884739, 0.16951563486244936, 0.22906242458375914, -0.101369412863761, -0.13697806444668004, -0.09092274044775195, -0.12286172523801042, 0.08688063379835853, 0.0, 0.0, 0.0, 0.11739972105573315, -0.18385640565554695, 0.09328684047460456, 0.09148382273949096};
            double[] numericGradient = tinyNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 5!");
            }
            Log.info("passed testTinyGradientNumeric on instance5");


            calculatedGradient = new double[]{0.04325721336684296, 0.06037745192699617, 0.14639508938429913, 0.2043349889291335, -0.10499964231414083, -0.14655615165892755, -0.033182774306084184, -0.046315771218274904, -0.2499058182614533, 0.0, 0.0, 0.0, -0.34881294741495594, -0.013728720382744086, 0.027466193763814317, -0.013727270431473926};
            numericGradient = tinyNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 55!");
            }
            Log.info("passed testTinyGradientNumeric on instance55");


            calculatedGradient = new double[]{0.08802751416681076, 0.10256381743900533, -0.13031782541972348, -0.1518376846476599, 0.10325106991615485, 0.1203012978123752, 0.1257718340585967, 0.1465409971146414, 0.15961883703674573, 0.0, 0.0, 0.0, 0.18597727580171863, -0.0815107625840028, -0.08245081950519761, 0.16240483180496312};
            numericGradient = tinyNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 123!");
            }
            Log.info("passed testTinyGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testTinyGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }



    /**
     * This tests calculation of the numeric gradient for
     * the small fully connected neural network generated
     * by PA11Tests.createSmallNeuralNetwork()
     */
    public static void testSmallGradientNumericSVM() {
        try {
            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(irisData, LossFunction.SVM);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[smallNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            smallNN.setWeights(weights);

            double[] calculatedGradient = new double[]{0.1107791480770004, 0.12212544198142439, 0.17778086225384016, -0.4037192313788296, -0.4450692192570216, -0.6478976799506597, 0.24142186694220413, 0.2661489328925626, 0.3874392329183962, 0.216542022091204, 0.23872082643983106, 0.3475114940254542, -0.2069153204331542, 0.023712248831131433, 0.03725935560083826, 0.029679343427346794, -0.2281081334931656, 0.055060009884755345, 0.0865164950791808, 0.06891565318412063, -0.3320623498392905, 0.011772748331750904, 0.018498667042621264, 0.01473531519025073, 0.13442194468638036, 0.14637776324377683, -0.1260245752199296, -0.06824042775299688, 0.21121891347064548, 0.0, 0.0, 0.0, 0.16824872162146676, 0.16046706852179682, -0.13815482313717098, -0.07480877584065126};
            double[] numericGradient = smallNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 5!");
            }
            Log.info("passed testSmallGradientNumeric on instance5");


            calculatedGradient = new double[]{-0.18299762816198495, -0.2305313928641084, -0.2852679170661787, -0.6193175938751239, -0.7801857893063868, -0.9654302235428958, 0.4441960876899742, 0.5595763452603819, 0.6924400897112548, 0.1403781735298537, 0.17684150477315086, 0.2188300995964454, 1.0572149045096069, -0.01824056572452548, -0.021056700827415398, -0.023849021646427104, 1.331827242934125, 0.089234808431371, 0.10301164032888721, 0.11667198096532161, 1.6480513909122863, -0.047555090976203473, -0.054897051349200865, -0.06217693138133029, -0.6488707315899234, 0.0054099991153577776, -0.010832801322635532, 0.005409096504038757, -0.7490489206674056, 0.0, 0.0, 0.0, -0.848380061757581, 0.006311509093137602, -0.012637954016625486, 0.006310455491487232};
            numericGradient = smallNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 55!");
            }
            Log.info("passed testSmallGradientNumeric on instance55");


            calculatedGradient = new double[]{-0.09812123114016913, -0.1246880421135188, -0.15406627085212676, 0.14526078118137775, 0.18459086481570353, 0.2280830213585716, -0.11509040387380765, -0.1462517040806688, -0.18071062868330046, -0.14019352567729015, -0.17815161790224465, -0.22012660139125728, -0.1779216063368949, -0.027927691270690502, -0.04441458312243185, -0.03285915317619015, -0.2260947717935835, -0.030452969035366095, -0.04843064171922151, -0.03583034757781434, -0.27936582158716305, -0.027596809282215418, -0.043888366274558166, -0.03246984459082114, 0.1224722989690008, 0.0732372162914885, 0.13358018247089376, -0.13565492951173042, 0.19477285562174984, 0.0, 0.0, 0.0, 0.1440984176426241, 0.07985329286697151, 0.14564749850620728, -0.1479096711776151};
            numericGradient = smallNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 123!");
            }
            Log.info("passed testSmallGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testSmallGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    /**
     * This tests calculation of the numeric gradient for
     * the large fully connected neural network generated
     * by PA11Tests.createLargeNeuralNetwork()
     */
    public static void testLargeGradientNumericSVM() {
        try {
            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(irisData, LossFunction.SVM);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[largeNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            largeNN.setWeights(weights);

            double[] calculatedGradient = new double[]{7.244249644600131E-4, 8.676670493201755E-4, 0.0013272283272414143, -0.002640067053860662, -0.0031620916995933612, -0.004836897549154173, 0.001578744912578145, 0.0018909140919731726, 0.0028924396211493786, 0.0014160472794344514, 0.0016960444160218913, 0.002594356951490795, -0.0013530943032691312, 9.007239398783895E-5, 1.4523937608146298E-5, 1.0931699989669141E-4, 6.9966255011877365E-6, 1.244937486433173E-4, -0.0016206447295274984, 2.0914603382493624E-4, 3.3721914149964505E-5, 2.5383473101214804E-4, 1.624922418841379E-5, 2.890732098137505E-4, -0.002479020322354586, 4.4719783431901305E-5, 7.2097883219157666E-6, 5.42743627818254E-5, 3.47499806707674E-6, 6.180833622693171E-5, 5.106071121474542E-4, 9.731548900049347E-5, 8.117950756059145E-5, 1.496536228273726E-4, 1.2569056906386322E-4, 8.233191906015236E-5, 1.6765699939469414E-4, 1.3985479441203097E-4, 2.578226521166016E-4, 2.1654011916893978E-4, 6.197065083313191E-4, 1.0542011708025711E-4, 8.79385453345094E-5, 1.6211587627878998E-4, 1.361577517400292E-4, 3.9670489115906094E-5, 1.7418511077949006E-4, 1.4529932812479274E-4, 2.6786128870526227E-4, 2.2497115281794322E-4, 7.057376905095225E-4, 1.1290302026623067E-4, 9.418132940197665E-5, 1.7362333792902973E-4, 1.458233533924158E-4, -1.7809975716431836E-4, -0.36458315477716496, 9.729772543209947E-5, 0.18229157738858248, -1.4856560426323995E-4, -0.39842193766403966, 1.0632827951440049E-4, 0.19921096883201983, -2.7388091794477987E-4, -0.3035050377686588, 8.099743098455292E-5, 0.15175251832921788, -2.3002488802603693E-4, -0.35367152428911197, 9.438450021548306E-5, 0.1768357626996675};
            double[] numericGradient = largeNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 5!");
            }
            Log.info("passed testLargeGradientNumeric on instance5");


            calculatedGradient = new double[]{-0.01508536295879992, -0.020612820339493965, -0.026778451678310944, -0.05105329492494093, -0.06975982791423974, -0.09062612771337797, 0.0366171948318339, 0.050034172360824414, 0.0650002074387146, 0.011572042213359168, 0.015812173792539852, 0.020541852663313875, 0.08715125510150301, -8.111000759924991E-4, -0.006159240895087237, -0.0010670153649527947, -0.0029677349466794567, -0.00132281297027248, 0.1190845055898393, 0.003967989270492467, 0.030131651618248156, 0.005219954468671517, 0.01451848197220329, 0.006471344571323812, 0.15470462244593364, -0.0021146251416581663, -0.0160577895425007, -0.0027818236603138757, -0.007737204210656046, -0.0034487135369687394, -0.02885322980539229, 2.511624241918753E-4, 0.008208020929600934, 1.2725043241346157E-4, 0.012708584185006089, -0.21910225189714083, -0.011388865406303239, -0.37218844672892715, -0.005770118827186366, -0.5762640187256807, -0.0379568931840879, 2.9484303887272745E-4, 0.009635474640390385, 1.493805079633148E-4, 0.014918726254364856, -0.1055711373876278, -0.01183226516765501, -0.38667885893239884, -0.005994766905104143, -0.5986997075879685, -0.04705636547086556, 3.38513661546358E-4, 0.01106267744077627, 1.7150725284409418E-4, 0.01712848085588803, 0.012098108070901503, -0.24850125956987767, 0.35297866740613415, -0.24850125956987767, 0.3953665972389331, 0.19921096883201983, -0.28296525678150886, 0.1992109677217968, 0.006129453611336544, -0.24933523801173862, 0.35416327426318617, -0.2493352369015156, 0.6121510409684561, 0.1768357626996675, -0.25118283786262907, 0.1768357626996675};
            numericGradient = largeNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 55!");
            }
            Log.info("passed testLargeGradientNumeric on instance55");


            calculatedGradient = new double[]{-6.827871601444713E-6, -9.423573033018329E-6, -1.2234657731369225E-5, 1.0109690862236675E-5, 1.3951062527439717E-5, 1.8109957977685553E-5, -8.00914889964588E-6, -1.1053380433168059E-5, -1.4348522370255523E-5, -9.756639940405876E-6, -1.3464784842653899E-5, -1.7477130853649214E-5, -1.2383427616668996E-5, -1.0946799022804043E-6, -0.02463493853355203, -1.325606291402437E-6, -0.011869976113132452, -1.4988010832439613E-6, -1.709077324107966E-5, -1.1945999744966684E-6, -0.026862481128375748, -1.4432899320127035E-6, -0.012943286442634871, -1.6364687382974807E-6, -2.2182256032010628E-5, -1.0791367799356522E-6, -0.024343069782162274, -1.3100631690576847E-6, -0.011729344162603184, -1.4832579608992091E-6, 4.7983839124299266E-6, -1.98507876802978E-6, -0.11353700868355077, -6.972200594645983E-7, -0.17579074196305555, 0.1080324762980922, 3.206324095117452E-6, 0.18351451469555968, 1.1302070390684094E-6, 0.2841377733098227, 5.804245972740318E-6, -2.136069099378801E-6, -0.12221130774037192, -7.505107646466058E-7, -0.18922126310272347, 0.05205383235562522, 3.33288951992472E-6, 0.19065928613670735, 1.1723955140041653E-6, 0.29520011102235344, 6.576961197879427E-6, -2.275957200481571E-6, -0.1301554264365734, -8.01581023779363E-7, -0.2015212441364156, -3.4061642395499803E-6, -0.24999914582934935, -0.17509184546682377, 0.4999982960995908, -0.19494293734823032, 0.19921096994224285, 0.13952134114703085, -0.3984219398844857, -1.2012613126444194E-6, -0.2499997386884445, -0.17509226069023498, 0.49999947515644294, -0.3018325456238813, 0.17683576158944447, 0.12385042325036011, -0.35367152317888895};
            numericGradient = largeNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 123!");
            }
            Log.info("passed testLargeGradientNumeric on instance123");


        } catch (Exception e) {
            Log.fatal("Failed testLargeGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumericSOFTMAX() {
        try {
            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(irisData, LossFunction.SOFTMAX);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[tinyNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            tinyNN.setWeights(weights);
            
            double[] calculatedGradient = new double[]{-0.016482892828406648, -0.022123242127136677, 0.06006961150717416, 0.08062508438655414, -0.03592129704799163, -0.048213354419957, -0.032219409362710394, -0.04324470448402451, 0.030787047355929076, 0.0, 0.0, 0.0, 0.041322195665216555, -0.061818110630440515, 0.032455584886292854, 0.02969098966687511};
            double[] numericGradient = tinyNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 5!");
            }
            Log.info("passed testTinyGradientNumeric on instance5");

            calculatedGradient = new double[]{0.014466190467743445, 0.020191185390316946, 0.048957823350903595, 0.06833289489804883, -0.035114252616708086, -0.049010725478026984, -0.011097068819410083, -0.015488735849444879, -0.08357414871262847, 0.0, 0.0, 0.0, -0.11664863852622886, -0.0045877279752915, 0.009184686344809734, -0.004593545543940536};
            numericGradient = tinyNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 55!");
            }
            Log.info("passed testTinyGradientNumeric on instance55");

            calculatedGradient = new double[]{0.028464075541023703, 0.03309854501587495, -0.042138829092763785, -0.04899979977324165, 0.03338667786678684, 0.03882263843735245, 0.040668861611692364, 0.04729049818408271, 0.05161343685777808, 0.0, 0.0, 0.0, 0.06001704577762723, -0.027572998329006282, -0.026440981626407734, 0.053509326969347626};
            numericGradient = tinyNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 123!");
            }
            Log.info("passed testTinyGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testTinyGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }



    /**
     * This tests calculation of the numeric gradient for
     * the small fully connected neural network generated
     * by PA11Tests.createSmallNeuralNetwork()
     */
    public static void testSmallGradientNumericSOFTMAX() {
        try {
            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(irisData, LossFunction.SOFTMAX);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[smallNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            smallNN.setWeights(weights);

            double[] calculatedGradient = new double[]{0.02930872544659735, 0.03230879452864599, 0.04703098577252263, -0.10681158402192636, -0.11774491381899566, -0.17139790364772978, 0.06387273598917886, 0.07041080829495172, 0.10249499915460092, 0.057290301391788034, 0.0631545893359231, 0.09193232619608693, -0.054743375388355275, 0.006246665407161345, 0.009856420124521037, 0.007875512464394774, -0.06034695632095577, 0.014504798473424785, 0.022886679218458994, 0.01828699858208438, -0.08784533767070002, 0.0031013680512614883, 0.004893554450546844, 0.003910066714851723, 0.03541160586983949, 0.04646166007482577, -0.03329371778271195, -0.025292239413943207, 0.05587489138036972, 0.0, 0.0, 0.0, 0.04464535963144556, 0.050933737227865095, -0.036498337685486604, -0.027726694273866315};
            double[] numericGradient = smallNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 5!");
            }
            Log.info("passed testSmallGradientNumeric on instance5");


            calculatedGradient = new double[]{-0.060635536591746586, -0.07638565668877106, -0.09452238325380335, -0.20520842025639752, -0.25851144780375535, -0.3198914422242183, 0.14718260721302556, 0.1854133879763964, 0.22943725364044099, 0.046513749518695136, 0.058595726049048835, 0.07250847922080084, 0.350303952512121, -0.006043918698850348, -0.006977052269263595, -0.007902294374417806, 0.44129564114037123, 0.02956750511096118, 0.03413249016759323, 0.03865887943099722, 0.5460752372510314, -0.015757140037209183, -0.018189916239919057, -0.020602121120205652, -0.21500004998564748, 0.0017936663265771813, -0.003589475383591889, 0.0017912671346209663, -0.24819433397382795, 0.0, 0.0, 0.0, -0.2811079535813832, 0.002092560569266766, -0.004187615809669865, 0.0020897594765756367};
            numericGradient = smallNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 55!");
            }
            Log.info("passed testSmallGradientNumeric on instance55");


            calculatedGradient = new double[]{-0.04068910541832338, -0.05170374461904714, -0.06388417794767065, 0.06023702536772646, 0.07654333611384345, 0.09457551075620074, -0.0477259143316644, -0.06064543645045717, -0.07493236475752951, -0.05813572401258682, -0.07387320866136804, -0.09127634870154111, -0.07378088029419416, -0.011545785438826783, -0.018416378422259072, -0.013656711317366899, -0.09375354603768926, -0.01258977921914095, -0.02008162525157786, -0.014891579080966721, -0.1158401230494377, -0.011408994859962718, -0.018198185181006465, -0.013494909634204078, 0.05063214736367172, 0.021637179825617636, 0.055227876716301694, -0.048081758574625155, 0.08076200375128906, 0.0, 0.0, 0.0, 0.059889263548384974, 0.023591830711211514, 0.06021703025105296, -0.05242534850680158};
            numericGradient = smallNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 123!");
            }
            Log.info("passed testSmallGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testSmallGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    /**
     * This tests calculation of the numeric gradient for
     * the large fully connected neural network generated
     * by PA11Tests.createLargeNeuralNetwork()
     */
    public static void testLargeGradientNumericSOFTMAX() {
        try {
            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(irisData, LossFunction.SOFTMAX);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[largeNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            largeNN.setWeights(weights);

            double[] calculatedGradient = new double[]{1.6858736628933002E-4, 2.019240330497496E-4, 3.0887292723491555E-4, -6.143974218275616E-4, -7.358835762971694E-4, -0.0011256445775487123, 3.6740610553920305E-4, 4.400546593785748E-4, 6.73129330053257E-4, 3.295430595073867E-4, 3.9470537949171103E-4, 6.037609301401403E-4, -3.1489311158594546E-4, 2.0961010704922955E-5, 3.380073998471289E-6, 2.5439650386260837E-5, 1.6286971771251046E-6, 2.897126982759346E-5, -3.771571943644858E-4, 4.8672732511079175E-5, 7.848166561075232E-6, 5.907219158274302E-5, 3.7808645103609706E-6, 6.727396417716136E-5, -5.769179578507533E-4, 1.0407230632836217E-5, 1.676991878696299E-6, 1.2630452239648093E-5, 8.093525849517391E-7, 1.438460461855584E-5, 1.1882828054865513E-4, 2.264799459084088E-5, 1.8891554987021664E-5, 3.482714117097885E-5, 2.9251046029799E-5, 1.9160784070493264E-5, 3.901712286591419E-5, 3.254729818991109E-5, 6.0002003365866585E-5, 5.039357819924817E-5, 1.4421908112183246E-4, 2.4533153286654397E-5, 2.046474101291551E-5, 3.772815393432438E-5, 3.168632023431428E-5, 9.231504449758177E-6, 4.053757329813834E-5, 3.381517288403302E-5, 6.233680238665329E-5, 5.235534228376082E-5, 1.642397329248979E-4, 2.6274538100778955E-5, 2.191802295214984E-5, 4.040601186972026E-5, 3.393618719371716E-5, -4.1447401066818657E-5, -0.11235727070957324, 2.2644108810254693E-5, 0.06993430667900924, -3.457401032136431E-5, -0.12278570882262585, 2.474465077284549E-5, 0.07642526000939398, -6.373790384373024E-5, -0.09353421037694432, 1.8848811400573595E-5, 0.058218308507385075, -5.353162357835117E-5, -0.10899452396184728, 2.1965762542208722E-5, 0.06784123818270871};
            double[] numericGradient = largeNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 5!");
            }
            Log.info("passed testLargeGradientNumeric on instance5");


            calculatedGradient = new double[]{-0.004559078670141048, -0.006229579074812364, -0.008092948533544586, -0.015429257871346636, -0.021082722234666562, -0.027388907253289574, 0.011066399463466325, 0.015121262020301174, 0.019644275650598786, 0.003497287459630627, 0.004778734075117086, 0.0062081312313111425, 0.02633873563695488, -2.451289171645499E-4, -0.0018614371155578624, -3.224714939520368E-4, -8.96903107339142E-4, -3.9977743337971106E-4, 0.03598956022088373, 0.0011992001835992028, 0.009106347342857646, 0.00157756752106053, 0.004387756824542066, 0.0019557611086185034, 0.04675462350611781, -6.390799001110281E-4, -0.004852963586543524, -8.407191609549614E-4, -0.002338327309558963, -0.001042265163064826, -0.00871998473428448, 7.590594819362195E-5, 0.002480617378175509, 3.845757046150311E-5, 0.0038407715896582317, -0.06621678982909884, -0.003441927298730718, -0.11248229569993384, -0.0017438378518974673, -0.17415774433704456, -0.011471282257424775, 8.910649995641506E-5, 0.0029120206246346925, 4.514555396184505E-5, 0.004508717288409514, -0.03190556763232877, -0.003575931772914487, -0.11686156986545626, -0.0018117313205223695, -0.1809382282891292, -0.01422131246098246, 1.0230427616164661E-4, 0.0033433467105936643, 5.183242723916237E-5, 0.005176545858631698, 0.0036562736172029986, -0.07510171928792886, 0.10667674354980505, -0.07510171817770583, 0.11948716627241396, 0.060205274099445205, -0.08551738261974862, 0.060205271878999156, 0.001852435982385714, -0.07535376322920229, 0.10703475439299837, -0.07535376211897926, 0.1850034719863558, 0.053443068859238, -0.07591214212787634, 0.05344306774901497};
            numericGradient = largeNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 55!");
            }
            Log.info("passed testLargeGradientNumeric on instance55");


            calculatedGradient = new double[]{-2.708944180085382E-6, -3.738120923912902E-6, -4.852784840636559E-6, 4.0101255649460654E-6, 5.534461777756405E-6, 7.182032746300138E-6, -3.176348073452573E-6, -4.385380947269368E-6, -5.6910032242285524E-6, -3.870237463843296E-6, -5.341282971471628E-6, -6.933342788784103E-6, -4.9116266609416925E-6, -4.340972026284362E-7, -0.009771381481726849, -5.25135490647699E-7, -0.004708192724578453, -5.961897642237091E-7, -6.777911565336581E-6, -4.729550084903167E-7, -0.010654929161191262, -5.728750807065808E-7, -0.00513391995582424, -6.494804694057166E-7, -8.798517470154366E-6, -4.285460875053104E-7, -0.00965561297583406, -5.195843755245733E-7, -0.004652411789152211, -5.873079800267078E-7, 1.9040324872321435E-6, -7.882583474838611E-7, -0.045034139750654845, -2.76445533131664E-7, -0.06972691313222867, 0.04285078403221121, 1.2723155862204294E-6, 0.07279052582909173, 4.4630965589931293E-7, 0.11270246513817028, 2.3037127760972E-6, -8.459899447643693E-7, -0.04847477752534246, -2.9753977059954195E-7, -0.07505409294061849, 0.02064700854731427, 1.3211653993039363E-6, 0.07562447890130386, 4.651834473179406E-7, 0.11709031078055432, 2.609024107869118E-6, -9.026113190202523E-7, -0.05162579141959611, -3.1752378504279477E-7, -0.07993284589780103, -1.3511414209688155E-6, -0.07541883895179069, -0.06944969710431792, 0.17458031020822773, -0.0773235775408665, 0.06009724495825708, 0.05534075531166138, -0.1391137260942088, -4.7628567756419216E-7, -0.07541901769769765, -0.06944986141732556, 0.17458072210096987, -0.11972104752011603, 0.053347172235262974, 0.04912492856945505, -0.12348859046440452};
            numericGradient = largeNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 123!");
            }
            Log.info("passed testLargeGradientNumeric on instance123");


        } catch (Exception e) {
            Log.fatal("Failed testLargeGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }

}

