import java.util.*;
/**
 * Encapsulation of a hidden node.
 * 
 * @author Andrew M. Elgert
 * @version 1.1
 */
public class HiddenNeuron
{
    private boolean biasNode;
    private HiddenLayer myLayer;
    private Vector<Double> weights, weightDeltas;
    private Double delta, value;
    
    /**
     * Options Constructor
     * 
     * @param myLayer Assigns neuron to a particular layer of type InputLayer
     * @param weights Weights associated with inputs from the previous layer, passed through this layer
     * @param biasNode Specifies whether this neuron is the bias node
     */
    public HiddenNeuron(HiddenLayer myLayer, Vector<Double> weights, boolean biasNode)
    {
        this.myLayer = myLayer;
        this.weights = weights;
        this.weightDeltas = new Vector<Double>();
        this.delta = 1.0;
        this.biasNode = biasNode;
        this.activate();
    }
    
    /**
     * Constructor
     * 
     * @param myLayer
     * @param biasNode
     */
    public HiddenNeuron(HiddenLayer myLayer, boolean biasNode)
    {
        this(myLayer, new Vector<Double>(), biasNode);
    }
    
    
    /**
     * Add weights to this node, to properly activate it
     * 
     * @param weights Vector of weights to be combined with inputs during activation
     * 
     * @return True on success
     */
    public boolean addWeights(Vector<Double> weights)
    {
        this.weights = weights;
        
        return true;
    }
    
    /**
     * Change the so that the node value is changed
     * 
     * @return True on success
     */
    
    public boolean activate()
    {
        if (biasNode)
            value = Constants.biasNodeValue;
        else
        {
            double summation;
            Vector<Double> inputValues;
        
            inputValues = myLayer.getPrevLayer().getValues();
        
            summation = 0.;
            
            for (int ii = 0; ii < weights.size(); ii++)
                summation += inputValues.get(ii) * weights.get(ii);
            
            value = 1 / (1 + Math.exp(-1 * summation));
        }
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
            weights.set(ii, weights.get(ii)+weightDeltas.get(ii));
        }
        weightDeltas.removeAllElements();
        return true;
    }
    
    /**
     * Compute each delta (small-delta) based on the values of deltas in the next layers, weights, and other variables
     * 
     * @param nextDeltas Deltas (small-deltas) from the next layer
     * @return nextWeights Weights of connections between this node and each node in the next layer (excluding potentially a bias node)
     */
    public boolean computeDelta(Vector<Double> nextDeltas, Vector<Double> nextWeights)
    {
        Double summation;
        
        summation = new Double(0.);
        
        for (int ii = 0; ii < nextDeltas.size(); ii ++)
            summation += nextDeltas.get(ii) * nextWeights.get(ii);
        
        this.delta = this.getValue() * (1 - this.getValue()) * summation;
        
        return true;
    }
    
    /**
     * Compute the offsets this node should apply to more thorougly achieve its goal
     * 
     * @return True on success
     */
    
    public boolean computeWeightDeltas()
    {
        Vector<Double> inputValues;
        
        inputValues = myLayer.getPrevLayer().getValues();
        
        for (int ii = 0; ii < weights.size(); ii ++)
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
}