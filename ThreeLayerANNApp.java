import java.util.*;
import java.io.*;

/**
 * UI class; lots of code, so there's probably an error somewhere...UI isn't my specialty...
 * 
 * @author Andrew Elgert 
 * @version 0.9
 */
public class ThreeLayerANNApp
{
    public static void main(String[] argv)
    {
        int numInput, numHidden, numOutput;
        Scanner configScanner, inScanner, trainScanner, testScanner, tempScanner;
        String[] splitLine;
        Vector<String> configData, inData, trainData, testData;
        Vector<Vector<Double>> hiddenWeights, outputWeights, inputs, outputs, expOutputs;
        Vector<Double> temp;
        
        Double rmsError;
        
        ThreeLayerANN ann;
        
        configData = new Vector<String>();
        inData = new Vector<String>();
        trainData = new Vector<String>();
        testData = new Vector<String>();
        
        inputs = new Vector<Vector<Double>>();
        outputs = new Vector<Vector<Double>>();
        expOutputs = new Vector<Vector<Double>>();
        
        rmsError = .0;
        temp = new Vector<Double>();
        
        try
        {
            configScanner = new Scanner(new File("net.cfg"));
            inScanner = new Scanner(new File("in.dat"));
            trainScanner = new Scanner(new File("train.dat"));
            testScanner = new Scanner(new File("test.dat"));
        
            while(configScanner.hasNextLine())
                configData.add(configScanner.nextLine());
        
            while(inScanner.hasNextLine())
                inData.add(inScanner.nextLine());
        
            while(trainScanner.hasNextLine())
                trainData.add(trainScanner.nextLine());
            
            while(testScanner.hasNextLine())
                testData.add(testScanner.nextLine());
        } catch(FileNotFoundException fe) {
            System.err.println(fe.toString());
            System.err.println("Could not find one of the specified input files. Exiting...");
            System.exit(1);
        }
        
        
        splitLine = configData.get(0).split("[ \t\r\n]");
        
        if (splitLine.length > 3)
        {
            System.out.println("The first line of net.cfg has too many parameters...  Please retry with only three parameters.");
            System.exit(1);
        }
        
        numInput = Integer.parseInt(splitLine[0]);
        numHidden = Integer.parseInt(splitLine[1]);
        numOutput = Integer.parseInt(splitLine[2]);
        
        if (numInput > 10 || numHidden > 10 | numOutput > 10)
        {
            System.out.println("No layer may have more than 10 nodes.  Please adjust net.cfg accordingly.");
            System.exit(1);
        }
        
        configData.remove(0);
        
        hiddenWeights = new Vector<Vector<Double>>();
        outputWeights = new Vector<Vector<Double>>();
        
        ann = new ThreeLayerANN(numInput, numHidden, numOutput);
        
        for(int ii = 0; ii < numHidden; ii++)
        {
            if (configData.get(0).split("[ \t\r\n]").length > numHidden + 1)
            {
                System.out.println("You have too many weights for a hidden node.");
                System.exit(1);
            }
            tempScanner = new Scanner(configData.get(0));
            temp = new Vector<Double>();
            for(int jj = 0; jj < numInput + 1; jj++)
            {
                temp.add(tempScanner.nextDouble());
            }
            hiddenWeights.add(ii, temp);
            configData.remove(0);
        }
        
        for(int ii = 0; ii < numOutput; ii++)
        {
            if(configData.get(0).split("[ \t\r\n]").length > numOutput+1)
            {
                System.out.println("You have too many weights specified on an output node.");
                System.exit(1);
            }
            
            tempScanner = new Scanner(configData.get(0));
            temp = new Vector<Double>();
            for(int jj = 0; jj < numHidden + 1; jj++)
            {
                temp.add(tempScanner.nextDouble());
            }
            outputWeights.add(ii,temp);
            configData.remove(0);
        }
        
        ann.setHiddenWeights(hiddenWeights);
        ann.setOutputWeights(outputWeights);
        
        
        for(String ss : inData)
        {
            tempScanner = new Scanner(ss);
            temp = new Vector<Double>();
            
            while(tempScanner.hasNext())
            {
                temp.add(tempScanner.nextDouble());
            }
            inputs.add(temp);
        }
        
        for(Vector<Double> vv : inputs)
        {
            ann.setInputValues(vv);
            ann.feedForward();
            outputs.add(ann.getOutput());
        }
        
        System.out.println("Part One");
        System.out.println();
        for(int ii = 0; ii < inputs.size(); ii++)
        {
            System.out.println("Inputs:\t" + inputs.get(ii) + "\tOutputs:\t" + outputs.get(ii));
        }
        
        inputs.removeAllElements();
        outputs.removeAllElements();
        
        for(String ss : trainData)
        {
            int ii;
            splitLine = ss.split("[ \t\r\n]");
            temp = new Vector<Double>();
            for(ii = 0; ii < numInput; ii ++)
            {
                temp.add(Double.parseDouble(splitLine[ii]));
            }
            inputs.add(temp);
            temp = new Vector<Double>();
            for (ii = numInput; ii < numInput + numOutput; ii ++)
            {
                temp.add(Double.parseDouble(splitLine[ii]));
            }
            expOutputs.add(temp);
        }
        
        for(int ii = 0; ii < Constants.numIterations; ii++)
        {
            for(int jj = 0; jj < expOutputs.size(); jj++)
            {
                ann.setInputValues(inputs.get(jj));
                ann.feedForward();
                ann.backPropagate(expOutputs.get(jj));
            }
        }
        
        inputs.removeAllElements();
        outputs.removeAllElements();
        expOutputs.removeAllElements();
        
        for(String ss : testData)
        {
            int ii;
            splitLine = ss.split("[ \t\r\n]");
            temp = new Vector<Double>();
            for(ii = 0; ii < numInput; ii ++)
            {
                temp.add(Double.parseDouble(splitLine[ii]));
            }
            inputs.add(temp);
            temp = new Vector<Double>();
            for (ii = numInput; ii < numInput + numOutput; ii ++)
            {
                temp.add(Double.parseDouble(splitLine[ii]));
            }
            expOutputs.add(temp);
        }
        
        for(Vector<Double> vv : inputs)
        {
            ann.setInputValues(vv);
            ann.feedForward();
            outputs.add(ann.getOutput());
        }

        Double summation = 0.;
        
        for(int ii = 0; ii < expOutputs.size(); ii++)
            for(int jj = 0; jj < expOutputs.get(0).size(); jj++)
                summation += Math.pow(expOutputs.get(ii).get(jj) - outputs.get(ii).get(jj),2);
        
        rmsError = (summation)/(expOutputs.size() * expOutputs.get(0).size());
        
        System.out.println();
        System.out.println("Part Two");
        System.out.println();
        
        for(int ii = 0; ii < inputs.size(); ii++)
        {
            System.out.println("Inputs: " + inputs.get(ii) + "\tExpected Outputs: " + expOutputs.get(ii) + "\tActual outputs: " + outputs.get(ii));
        }

        System.out.println();
        System.out.println("\tRMS Error: " + rmsError);
        
        System.out.println();
        System.out.println("net.cfg:");
        
        hiddenWeights.removeAllElements();
        outputWeights.removeAllElements();
        
        hiddenWeights = ann.getHiddenWeights();
        hiddenWeights.remove(hiddenWeights.lastElement());
        outputWeights = ann.getOutputWeights();
        System.out.println();
        
        System.out.println(numInput + " " + numHidden + " " + numOutput);
        for(Vector<Double> vv : hiddenWeights)
        {
            for(Double dd : vv)
            { 
                System.out.print(dd + " ");
            }
            System.out.println();
        }
        
        for(Vector<Double> vv : outputWeights)
        {
            for (Double dd : vv)
            {
                System.out.print(dd + " ");
            }
            System.out.println();
        }
    }
}













