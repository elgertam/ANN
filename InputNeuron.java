
/**
 * Encapsulates a neuron in the input layer
 * 
 * @author Andrew Elgert
 * @version 1.0
 */

public class InputNeuron
{
    private boolean biasNode;
    private InputLayer myLayer;
    private Double value;
    
    /**
     * Constructor with options
     * 
     * @param myLayer assigns neuron to a particular layer of type InputLayer
     * @param biasNode Specifies whether this neuron is the bias node
     */
    public InputNeuron(InputLayer myLayer, boolean biasNode)
    {
        this.myLayer = myLayer;
        if(biasNode)
            this.value = Constants.biasNodeValue;
        else
            this.value = 0.;
    }
    
    
    /**
     * Get the value of this neuron
     * 
     * @return Value of this neuron
     */
    public Double getValue()
    {
        return this.value;
    }
    
    /**
     * Set the value of this neuron
     * 
     * @param Value to be set
     * @return The previous value
     */
    public Double setValue(Double value)
    {
        Double temp = this.value;
        if(!biasNode)
        {
            this.value = value;
        }
        else
        {
            value = 1.;
        }
        return temp;
    }

}
