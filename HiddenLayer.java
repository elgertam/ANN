import java.util.*;
/**
 * Encapsulation of a hidden layer.
 * 
 * @author Andrew Elgert
 * @version 1.0
 */
public class HiddenLayer
implements Layer
{
    private boolean finalized;
    private Vector<HiddenNeuron> hiddenNeurons;
    private Layer nextLayer, prevLayer;
    
    /** Options constructor
     * 
     * @param hiddenNeurons Vector of hiddenNeurons
     * @param prevLayer A reference to the previous layer, used for message passing;
     * also allows for more extensability in the future
     * @param nextLayer A reference to the previous layer, used for message passing;
     * also allows for more extensability in the future
     */
    public HiddenLayer(Vector<HiddenNeuron> hiddenNeurons, Layer prevLayer, Layer nextLayer)
    {
        this.hiddenNeurons = hiddenNeurons;
        this.prevLayer = prevLayer;
        this.nextLayer = nextLayer;
        this.finalized = false;
    }
    
    /**Constructor
     * @param nextLayer, prevLayer
     */
    public HiddenLayer(Layer nextLayer, Layer prevLayer)
    {
        this(new Vector<HiddenNeuron>(), nextLayer, prevLayer);
    }
    
    /**
     * Default constructor
     */
    public HiddenLayer()
    {
        this(null, null);
    }

    /**
     * Activate every neuron in this layer
     * 
     * @return True on success
     */
    public boolean activate()
    {
        for(HiddenNeuron nn : hiddenNeurons)
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
        if (!finalized)
            hiddenNeurons.add(new HiddenNeuron(this, false));
        else
            return false;

        return true;
    }
    
    
    /**
     * 
     *Add "bias node" which affects the calcuations as a kind of control.  Has some interesting properties
     * 
     * @return True on success
     */
    public boolean addBiasNode()
    {
        hiddenNeurons.add(new HiddenNeuron(this, true));
        
        finalized = true;
        
        return true;
    }
    
    /**
     * Apply weight deltas for back-propagation
     * 
     * @return True on success
     */
    public boolean applyWeightDeltas()
    {
        for(HiddenNeuron nn : hiddenNeurons)
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
        Vector<Double> nextWeights, nextDeltas;
        nextDeltas = nextLayer.getDeltas();
        
        for(int ii = 0; ii < hiddenNeurons.size(); ii++)
        {
            nextWeights = nextLayer.getWeights(ii);
            hiddenNeurons.get(ii).computeDelta(nextDeltas, nextWeights);
        }
        
        return true;
    }
    
    /**
     * Compute Deltas (capital-delta) to apply to each weight based on the error gradient and other values
     * 
     * @return True on success
     */
    public boolean computeWeightDeltas()
    {
        for(HiddenNeuron nn : hiddenNeurons)
            nn.computeWeightDeltas();
        return true;
    }
    
    
     /**
     * Get the neurons in this layer - the hidden neurons
     * 
     * @return Vector of hidden neurons
     */
    public Vector<HiddenNeuron> getHiddenNeurons()
    {
        return hiddenNeurons;
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
        
        for(HiddenNeuron nn : hiddenNeurons)
            result.add(nn.getDelta());
            
        return result;
    }
    
    
    
    /**
     * Useful, allows for message passing/receving
     * 
     * @return Interface type allowing for potentially multiple hidden layers
     */    
    public Layer getNextLayer()
    {
        return nextLayer;
    }
    
    public Layer getPrevLayer()
    {
        return prevLayer;
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
        
        for (HiddenNeuron nn : hiddenNeurons)
            result.add(nn.getValue());
        
        return result;
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
        for(int ii = 0; ii < hiddenNeurons.size() - 1; ii++)        //bias node has no weight vector
        {
            hiddenNeurons.get(ii).addWeights(weightsArray.get(ii));
        }
        return true;
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
        Vector<Vector<Double>> result;
        result = new Vector<Vector<Double>>();
        for(HiddenNeuron nn : hiddenNeurons)
        {
            result.add(nn.getWeights());
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
        
        for(HiddenNeuron nn : hiddenNeurons)
            result.add(nn.getWeights().get(weightIndex));
            
        return result;
    }
    
    /**
     * Link this layer with the previous/next layer
     * 
     * @ return True on success
     */
    public boolean setNextLayer(Layer nextLayer)
    {
        this.nextLayer = nextLayer;
        
        return true;
    }
    
    public boolean setPrevLayer(Layer prevLayer)
    {
        this.prevLayer = prevLayer;
        
        return true;
    }
}

