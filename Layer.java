import java.util.*;
/**
 * Interface different layer types can use to facilitate messaging
 * 
 * @author Andrew Elgert
 * @version 0.8
 */

public interface Layer
{
    /**
     * Get all delta (small-delta) values in this layer
     * 
     * @return Vector of small-deltas needed for back-propagation
     */
    public Vector<Double> getDeltas();
   
    /**
     * Useful, allows for message passing/receving
     * 
     * @return Interface type allowing for potentially multiple hidden layers
     */   
    
    public Layer getPrevLayer();
    
    public Layer getNextLayer();
    
     /**
     * Get values of each neuron in this layer
     * 
     * @return Vector of the neuron values
     */
    public Vector<Double> getValues();
    
    /**
     * Get the weightIndex-th weight of each neuron, used in back-propagation
     * 
     * @return Vector of weights for connections from a particular node in the previous layer
     */
    public Vector<Double> getWeights(int weightIndex);
}
