import java.util.*;
/**
 * Encapsulation of an output layer
 * 
 * @author Andrew Elgert
 * @version 1.0
 */
public class OutputLayer
implements Layer
{
    private boolean finalized;
    private Vector<OutputNeuron> outputNeurons;
    private Layer nextLayer, prevLayer;
    
    
    /**
     * Constructor with options
     * 
     * @param outputNeurons vector of neurons used to create the class; not very useful
     * @param prevLayer A reference to the previous layer, used for message passing;
     * also allows for more extensability in the future
     */
    public OutputLayer(Vector<OutputNeuron> outputNeurons, Layer prevLayer)
    {
        this.outputNeurons = outputNeurons;
        this.nextLayer = null;
        this.prevLayer = prevLayer;
        this.finalized = false;
    }
    
    
    /**
     * Default constructor
     */
    public OutputLayer()
    {
        this(new Vector<OutputNeuron>(), null);
    }
    
    
    /**
     * Activate every neuron in this layer
     * 
     * @return True on success
     */
    public boolean activate()
    {
        for(OutputNeuron nn : outputNeurons)
            nn.activate();
        return true;
    }
    
    
    /**
     * Add a "blank slate" neuron to this layer; the neuron will be configured later
     * 
     * @return True on success
     */
    public boolean addNeuron()
    {
        outputNeurons.add(new OutputNeuron(this));
        return true;
    }
    
    
    /**
     * Apply weight deltas for back-propagation
     * 
     * @return True on success
     */
    public boolean applyWeightDeltas()
    {
        for(OutputNeuron nn : outputNeurons)
            nn.applyWeightDeltas();
            
        return true;
    }
    
    /**
     * Compute error gradients (deltas) to help "teach" the ANN
     * 
     * @return true on success
     */
    public boolean computeDeltas()
    {   
        for(OutputNeuron nn : outputNeurons)
            nn.computeDelta();
        
        return true;
    }
    
    
    /**
     * Compute Deltas (capital-delta) to apply to each weight based on the error gradient and other values
     * 
     * @return True on success
     */
    public boolean computeWeightDeltas()
    {
        for(OutputNeuron nn : outputNeurons)
        {
            nn.computeWeightDeltas();
        }
        
        return true;
    }
    
    
    /**
     * Get all delta (small-delta) values in this layer
     * 
     * @return Vector of small-deltas needed for back-propagation
     */
    public Vector<Double> getDeltas()
    {
        Vector<Double> result;
        result = new Vector<Double>();
        
        for(OutputNeuron nn : outputNeurons)
        {
            result.add(nn.getDelta());
        }
        
        return result;
    }
    
    
    /**
     * Get the neurons in this layer - the output neurons
     * 
     * @return Vector of output neurons
     */
    public Vector<OutputNeuron> getOutputNeurons()
    {
        return outputNeurons;
    }
    
    
    /**
     * Get values of each neuron in this layer
     * 
     * @return Vector of the neuron values
     */
    public Vector<Double> getValues()
    {
        Vector<Double> result;
        result = new Vector<Double>();
        
        for (OutputNeuron nn : outputNeurons)
            result.add(nn.getValue());
        
        return result;
    }
    
    /**
     * DO NOT USE;  Only implemented to allow for implementing the Layer interface
     * 
     * @return Interface type allowing for potentially multiple hidden layers
     */
    
    public Layer getNextLayer()
    {
        return nextLayer;
    }
    
    
    /**
     * Useful, allows for message passing/receving
     */
    public Layer getPrevLayer()
    {
        return prevLayer;
    }
    
    
    /**
     * Get the weights for this layer
     * 
     * @return 2-D vector with the weights in this layer
     * 
     * WARNING - MAY CONTAIN BUGS
     */
    public Vector<Vector<Double>> getWeights()
    {
        Vector<Vector<Double>> result = new Vector<Vector<Double>>();
        for(int ii = 0; ii < outputNeurons.size() - 1; ii++)
        {
            result.add(outputNeurons.get(ii).getWeights());
        }
        
        return result;
    }
    
    
    /**
     * Get the weightIndex-th weight of each neuron, used in back-propagation
     * 
     * @return Vector of weights for connections from a particular node in the previous layer
     */
    public Vector<Double> getWeights(int weightIndex)
    {
        Vector<Double> result;
        result = new Vector<Double>();
        
        for(OutputNeuron nn : outputNeurons)
            result.add(nn.getWeights().get(weightIndex));
        return result;
    }
    
    
    /**
     * Give the neural network the expected output, used for back-propagation training
     * 
     * @return True on success
     */
    public boolean setExpectedOutputs(Vector<Double> expectedOutputs)
    {
        for(int ii = 0; ii < outputNeurons.size(); ii++)
            outputNeurons.get(ii).setExpectedOutput(expectedOutputs.get(ii));
        return true;
    }
    
    /**
     * Link this layer with the previous layer
     * 
     * @return True on success
     */
    public boolean setPrevLayer(Layer prevLayer)
    {
        this.prevLayer = prevLayer;
        
        return true;
    }
    
    
    /**
     * Set weights for each node in this layer
     * 
     * @return True on success
     * 
     * WARNING - MAY CONTAIN BUGS
     */
    public boolean setWeights(Vector<Vector<Double>> weightsArray)
    {
        for(int ii = 0; ii < outputNeurons.size(); ii++)
            outputNeurons.get(ii).addWeights(weightsArray.get(ii));
        return true;
    }
}
