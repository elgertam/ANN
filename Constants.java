
/**
 * Static class specifying three constants to be used throughout this application
 * 
 * @author Andrew M. Elgert
 * @version 1.0.1.2.1.2.3.4.5
 */
public class Constants
{
    /**
     * Specifies the learning rate globally so that the code is more flexible
     */
    public static final Double learningRate = .5;
    /**
     * Specifies the bias node value (generally 1, but the world is an interesting
     * place
     */
    public static final Double biasNodeValue = 1.;
    /**
     * Used for the training iterations; can be modified for different performance
     */
    public static final Integer numIterations = 1000;
}
