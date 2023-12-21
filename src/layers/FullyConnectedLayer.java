package layers;
import java.util.ArrayList;
import java.util.Random;
import java.util.List;

public class FullyConnectedLayer extends Layer{
    //now for a set of weights
    private double[][] _weights;
    private int _inputLength;
    private int _outputLength;
    private double learningRate;
    private long SEED;
    private double[] lastZ;
    private double[] lastX;
    private final double leakyRelu = 0.01;
    public FullyConnectedLayer(int inputLength, int outputLength, long SEED, double learningRate){
        this._inputLength = inputLength;
        this._outputLength = outputLength;
        this.learningRate = learningRate;
        this.SEED = SEED;
        _weights = new double[_inputLength][_outputLength];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input){
        lastX = input;

        double[] z = new double[_outputLength];
        double[] out = new double[_outputLength];

        //store the input for backpropagation
        //move through our nodes
        //multiply the input by the weights

        for (int i = 0; i < _inputLength; i++){
            for (int j = 0; j < _outputLength; j++){
                z[j] += input[i] * _weights[i][j];
            }
        }
        lastZ = z;
        //store the z values for backpropagation
        //now we need to apply the activation function
        for (int i = 0; i < _inputLength; i++){
            for (int j = 0; j < _outputLength; j++){
                out[j] = ReLu(z[j]);
            }
        }

        return out;
    }
    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);

    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if(_nextLayer != null){
            return _nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {
        //chain rule. dLdO is the derivative of the loss function with respect to the output
        double[] dLdX = new double[_inputLength];

        double dodz;
        double dzdw;
        double dLdw;
        double dzdx;

        for(int k = 0; k < _inputLength; k++){
            double dLdX_Sum = 0;
            for(int j = 0; j < _outputLength; j++){
                dodz = derivativeReLu(lastZ[j]);
                //derivative of relu func
                dzdw = lastX[k];
                //derivative of z with respect to w stored in LastX
                dzdx = _weights[k][j];

                dLdw = dLdO[j]* dodz * dzdw;

                _weights[k][j] -= learningRate * dLdw;

                dLdX_Sum += dLdO[j] * dodz * dzdx;
            }
            dLdX[k] = dLdX_Sum;
        }
        if(_previousLayer != null) {
            _previousLayer.backPropagation(dLdX);
        }

    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outputLength;
    }

    public void setRandomWeights(){
        Random random = new Random(SEED);
        for (int i = 0; i < _inputLength; i++){
            for (int j = 0; j < _outputLength; j++){
                _weights[i][j] = random.nextGaussian();
                //Gaussian is a number that is equally distrubuted around 0
                //that way we get no extreme values.
            }
        }
    }

    public double ReLu(double input){
        if(input <=0){
            return 0;
        } else {
            return input;
        }
    }

    public double derivativeReLu(double input){
        if(input <=0){
            return leakyRelu;
        } else {
            return 1;
        }
    }
}
