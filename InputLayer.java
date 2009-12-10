import java.util.*;

/**
 * Encapsulation of an input layer
 * 
 * @author Andrew Elgert
 * @version 1.1
 */

public class InputLayer
implements Layer
{
    private boolean finalized;
    private Layer nextLayer, prevLayer;
    private Vector<InputNeuron> inputNeurons;
    
    
    /**
     * Options constructor
     * 
     * @param inputNeurons Vector of neurons
     * @param nextLayer Hiddenlayer to which this layer sends output
     */
    public InputLayer(Vector<InputNeuron> inputNeurons, Layer nextLayer)
    {
        this.finalized = false;
        this.inputNeurons = inputNeurons;
        this.nextLayer = nextLayer;
        this.prevLayer = null;
    }
    
    /**
     * Constructor
     *
     * @param nextLayer
     */
    
    public InputLayer(Layer nextLayer)
    {
        this(null, nextLayer);
    }
    
    /**
     * Constructor
     * 
     * *param inputNeurons
     */
    public InputLayer(Vector<InputNeuron> inputNeurons)
    {
        this(inputNeurons, null);
    }
    
    /**
     * Default Constructor
     */
    public InputLayer()
    {
        this(new Vector<InputNeuron>());
    }
    
    /**
     * Add a bias node and prevent further addition to the inputLayer
     * 
     * @return True on success
     */
    
    public boolean addBiasNode()
    {
        inputNeurons.add(new InputNeuron(this, true));
        
        finalized = true;
        
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
            inputNeurons.add(new InputNeuron(this, false));
        else
            return false;
        return true;
    }
    
    
    /**
     * Useless method; only implemented to meet Layer requirements
     * 
     * @return Null
     */
    public Vector<Double> getDeltas()
    {
        return null;
    }
    
    
    /**
     * Get the neurons in this layer
     *
     * @return vector of input neurons
     */
    public Vector<InputNeuron> getInputNeurons()
    {
        return inputNeurons;
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
        
        for (InputNeuron nn : inputNeurons)
            result.add(nn.getValue());
        
        return result;
    }
    
    
    /**
     * Useless method; implements Layer interface
     */
    public Vector<Double> getWeights(int weightIndex)
    {
        return new Vector<Double>();
    }
    
  
    /**
     * Link this layer with the next layer
     * 
     * @ return True on success
     */
    
    public boolean setNextLayer(Layer nextLayer)
    {
        this.nextLayer = nextLayer;
        
        return true;
    }
    
    
    /**
     * Allows the user to specify a vector of inputs
     * 
     * @param Vector of doubles that are the inputs to the ANN
     * 
     * @return True on success
     */
    public boolean setValues(Vector<Double> values)
    {
        for(int ii = 0; ii < inputNeurons.size() - 1; ii++)     //bias node has no weight vecto
            inputNeurons.get(ii).setValue(values.get(ii));
        return true;
    }
}
