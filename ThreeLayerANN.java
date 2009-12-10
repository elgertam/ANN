import java.util.*;
/**
 * Container for the "controller" part of the application.  The UI
 * should almost exclusively work through objects of this class;
 * 
 * @author Andrew Elgert
 * @version 1.0
 */
public class ThreeLayerANN
{
    private InputLayer inputLayer;
    private HiddenLayer hiddenLayer;
    private OutputLayer outputLayer;
    
    
    /**
     * Constructor creating a three-layer neural network
     * 
     * @param input integer specifying number of input nodes
     * @param hidden integer specifiying number of hidden nodes
     * @param output integer specifying number of output nodes
     */
    public ThreeLayerANN(int input, int hidden, int output)
    {
        inputLayer = new InputLayer();
        hiddenLayer = new HiddenLayer();
        outputLayer = new OutputLayer();
        
        inputLayer.setNextLayer(hiddenLayer);
        hiddenLayer.setPrevLayer(inputLayer);
        hiddenLayer.setNextLayer(outputLayer);
        outputLayer.setPrevLayer(hiddenLayer);
        
        for(int ii = 0; ii < input; ii ++)
            inputLayer.addNeuron();
        inputLayer.addBiasNode();
            
        for(int ii = 0; ii < hidden; ii ++)
            hiddenLayer.addNeuron();
        hiddenLayer.addBiasNode();
            
        for(int ii = 0; ii < output; ii ++)
            outputLayer.addNeuron();
    }
    
    /**
     * Perform back-propagation learning on the neural network.  The controller
     * need only invoke the method with expected outputs for it to begin
     * 
     * @param expectedOutputs vector holding the expected output for each node
     * 
     * @return true for success, false for failure
     */
    
    public boolean backPropagate(Vector<Double> expectedOutputs)
    {
        outputLayer.setExpectedOutputs(expectedOutputs);
        
        outputLayer.computeDeltas();
        outputLayer.computeWeightDeltas();
        
        hiddenLayer.computeDeltas();
        hiddenLayer.computeWeightDeltas();
        
        outputLayer.applyWeightDeltas();
        hiddenLayer.applyWeightDeltas();
        
        return true;
    }
    
    
    /**
     * Perform feed-forward operation, where each layer's values are recomputed based
     * on previous values
     * 
     * @return Output values of the neural network
     */
    public Vector<Double> feedForward()
    {
        hiddenLayer.activate();
        outputLayer.activate();
        
        return outputLayer.getValues();
    }
    
    
    /**
     * Returns the value of the nodes in the output layer
     * 
     * @return The values of all nodes in the output layer
     */
    public Vector<Double> getOutput()
    {
        return outputLayer.getValues();
    }
    
    
    /**
     * Returns the weights from output layer
     * 
     * @result 2-D vector of weights, representing node and weight dimensions
     */
    public Vector<Vector<Double>> getOutputWeights()
    {
        return outputLayer.getWeights();
    }
    
    
    /**
     * Return weights from the hidden layer
     * 
     * @result 2-D vector of weights, representing node and weight dimensions
     */
    public Vector<Vector<Double>> getHiddenWeights()
    {
        Vector<Vector<Double>> result;
        result = hiddenLayer.getWeights();
        return result;
    }
    
    
    /**
     * Set input values
     * 
     * @param inputValues Vector containing the input values to the network
     * 
     * @return True on success
     */
    public boolean setInputValues(Vector<Double> inputValues)
    {
        inputLayer.setValues(inputValues);
        
        return true;
    }
    
    
    /**
     * Set weights for the hidden layer
     * 
     * @param weightsArray 2-D Vector of weights to be assigned to the hidden layer
     * 
     * @return True on success
     */
    public boolean setHiddenWeights(Vector<Vector<Double>> weightsArray)
    {
        hiddenLayer.setWeights(weightsArray);
        
        return true;
    }
    
    public boolean setOutputWeights(Vector<Vector<Double>> weightsArray)
    {
        outputLayer.setWeights(weightsArray);
        
        return true;
    }
}