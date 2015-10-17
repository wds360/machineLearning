package randomizedOptomization;
import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import func.nn.activation.*;

import shared.DataSet;
import shared.DataSetDescription;
import shared.reader.CSVDataSetReader;
import shared.reader.DataSetReader;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying credit applications as either 
 * success or failure. 
 *
 * @author Michael Walton adapted from Adam Acosta and abalone ABAGAIL example
 * @version 1.0
 */
public class OptimizeNeualNet {

    private static Instance[] trainInstances = initializeInstances("digitsTrain.csv", "digitsTrainLabels.csv");
    private static Instance[] testInstances = initializeInstances("digitsTest.csv", "digitsTestLabels.csv");

    private static int inputLayer = 64, hiddenLayer = 20, outputLayer = 10;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(trainInstances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static Instance[] initializeInstances(String dataFile, String labelFile) {
        DataSetReader dsr = new CSVDataSetReader(new File("").getAbsolutePath() + "/data/" + dataFile);
        DataSetReader lsr = new CSVDataSetReader(new File("").getAbsolutePath() + "/data/" + labelFile);
        DataSet ds;
        DataSet labs;

        try {
            ds = dsr.read();
            labs = lsr.read();
            Instance[] instances = ds.getInstances();
            Instance[] labels = labs.getInstances();
            
            for(int i = 0; i < instances.length; i++) {
                instances[i].setLabel(new Instance(labels[i].getData()));
            }

            return instances;
        } catch (Exception e) {
            System.out.println("Failed to read input file");
            return null;
        }
    }

    public static void main(String[] args) {

        int it = args.length > 0 ? Integer.parseInt(args[0]): 1000;

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer},
                new LogisticSigmoid());
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new RandomizedHillClimbing(nnop[2]);//new StandardGeneticAlgorithm(100, 50, 5, nnop[2]);

        for(int i = 0; i < oa.length; i++) {

            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            for(int j = 0; j < it; j++) {
                oa[i].train();
            }
            end = System.nanoTime();

            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double trainErr = 0;
            start = System.nanoTime();
            for(int j = 0; j < trainInstances.length; j++) {
                networks[i].setInputValues(trainInstances[j].getData());
                networks[i].run();

                trainErr += measure.value(new Instance(networks[i].getOutputValues()), trainInstances[j]);
                
                //predicted = Double.parseDouble(trainInstances[j].getLabel().toString());
                //actual = Double.parseDouble(networks[i].getOutputValues().toString());

                //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;

            results +=  oaNames[i] + "," + it + "," + df.format(trainErr / trainInstances.length) + ","; 

            correct = 0;
            incorrect = 0;
            
            double testErr = 0;
            
            start = System.nanoTime();
            for(int j = 0; j < testInstances.length; j++) {
                networks[i].setInputValues(testInstances[j].getData());
                networks[i].run();

                //predicted = Double.parseDouble(testInstances[j].getLabel().toString());
                //actual = Double.parseDouble(networks[i].getOutputValues().toString());

                //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                
                testErr += measure.value(new Instance(networks[i].getOutputValues()), testInstances[j]);

            }
            end = System.nanoTime();
            testingTime += end - start;
            testingTime /= 2.0;
            testingTime /= Math.pow(10, 9);

            results += df.format(testErr / testInstances.length) + ","
                     + df.format(trainingTime) + "," + df.format(testingTime) + "\n";

        }

        System.out.print(results);
    }

}