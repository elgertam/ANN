import java.util.*;
/**
 * Encapsulates the behavior of an output neuron
 * 
 * @author Andrew Elgert
 * @version 1.1
 */
public class OutputNeuron
{
    private OutputLayer myLayer;
    private Vector<Double> weights, weightDeltas;
    private Double delta, value, expectedOutput;
    
    /**
     * Options Constructor
     * @param myLayer Binds this neuron with an OutputLayer
     * @param weights Weights associated with input from the previous layer
     * @param expectedOutput probably a pretty dumb attribute to have in a constructor... c'est la vie
     */
    
    public OutputNeuron(OutputLayer myLayer, Vector<Double> weights, Double expectedOutput)
    {
        this.myLayer = myLayer;
        this.weights = weights;
        this.expectedOutput = expectedOutput;
        this.weightDeltas = new Vector<Double>();
        this.delta = 1.0;
        this.activate();
    }
    
    
    /**
     * "Default"ish Constructor
     * 
     * @param myLayer
     */
    public OutputNeuron(OutputLayer myLayer)
    {
        this(myLayer, new Vector<Double>(), 0.);
    }
    
    /**
     * Change the so that the node value is changed
     * 
     * @return True on success
     */
    
    public boolean activate()
    {
        Double summation;
        Vector<Double> hiddenValues;
        
        hiddenValues = myLayer.getPrevLayer().getValues();
        
        summation = 0.;
        
        for (int ii = 0; ii < weights.size(); ii++)
            summation += hiddenValues.get(ii) * weights.get(ii);
        
        value = 1 / (1 + Math.exp(-1.0 * summation));
        return true;
    }
    
        
    /**
     * Apply Deltas(large-delta) to each weight based on the back-propagation algorithm
     * 
     * @return True on success
     */
    
    public boolean applyWeightDeltas()
    {
        for(int ii = 0; ii < weights.size(); ii ++)
        {
            weights.set(ii, weights.get(ii) + weightDeltas.get(ii));
            weightDeltas.set(ii, 0.);
        }
        return true;
    }
    
    public boolean addWeights(Vector<Double> weights)
    {
        this.weights = weights;
        
        return true;
    }
    
    public boolean computeDelta()
    {
        this.delta = value * (1. - value) * (expectedOutput - value);
        
        return true;
    }
    
    /**
     * Compute each delta (small-delta) based on the values of deltas in the next layers, weights, and other variables
     * 
     * @param nextDeltas Deltas (small-deltas) from the next layer
     * @return nextWeights Weights of connections between this node and each node in the next layer (excluding potentially a bias node)
     */
    
    public boolean computeWeightDeltas()
    {
        Vector<Double> inputValues;
        
        inputValues = myLayer.getPrevLayer().getValues();
        
        for (int ii = 0; ii < weights.size(); ii++)
            weightDeltas.add(ii, Constants.learningRate * inputValues.get(ii) * this.getDelta());
            
        return true;
    }
    
     /**
     * Get the value of this neuron
     * 
     * @return Value of this neuron
     */
    
    public Double getValue()
    {
        return value;
    }
    
     /**
     * Return the value of this delta
     * 
     * @return the delta (small-delta) for this node
     */
    
    public Double getDelta()
    {
        return delta;
    }

     /**
     * Get the weights associated with input
     * 
     * @return Vector of weights associated with each input value
     */
    
    public Vector<Double> getWeights()
    {
        return weights;
    }
    
    public boolean setExpectedOutput(Double expectedOutput)
    {
        this.expectedOutput = expectedOutput;
        
        return true;
    }
}