package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;



public class ConvolutionLayer extends Layer{

    //a convolution takes and input, and applies a filter to it
    //the filter is a matrix of weights
    //the filter is applied to the input, and the result is the output
    //the filter is applied to the input by multiplying the filter by a section of the input
    private  List<double[][]> _filters;
    private int _filterSize;
    private int _stepSize;
    private int _inLength;
    private int _inRows;
    private int _inCols;
    private long SEED;
    private double _learningRate;
    private List<double[][]> _lastInput;

    public ConvolutionLayer(int _filterSize, int _stepSize, int _inLength, int _inRows, int _inCols, long SEED, int numFilters, double _learningRate) {

        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
        this.SEED = SEED;
        this._learningRate = _learningRate;

        generateRandomFilters(numFilters);
    }
    private void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for (int n = 0; n<numFilters; n++){
            double[][] newFilter = new double[_filterSize][_filterSize];
            for (int i = 0; i<_filterSize; i++){
                for (int j = 0; j<_filterSize; j++){
                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }
            filters.add(newFilter);
        }
        _filters = filters;
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        //this is the forward pass for a convolutional layer
        _lastInput = list;
        List<double[][]> output = new ArrayList<>();
        //this is the output of the layer
        for(int m = 0; m < list.size(); m++){
            for (double[][] filter : _filters){
                output.add(convolve(list.get(m), filter, _stepSize));
            }

        }
        return output;
    }
    private double[][] convolve(double[][] input, double[][] filter, int stepsize){
        int outRows = (input.length - filter.length)/stepsize + 1;
        int outCols = (input[0].length - filter[0].length)/stepsize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];
        int outRow = 0;
        int outCol;
        //move filter across image

        for (int i = 0; i <= inRows - fRows; i+=stepsize){
            outCol = 0;
            for (int j = 0; j <= inCols - fCols; j+=stepsize){
                //apply filter to section of image
                double sum = 0.0;
                for (int x = 0; x < fRows; x++){
                    for (int y = 0; y < fCols; y++){
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                        sum += value;

                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;

        }
        return output;
    }
    public double[][] spaceArray(double[][] input){
        if(_stepSize == 1){
            return input;
        }
        int outRows = (input.length - 1)*_stepSize + 1;
        //the number of rows in the output
        int outCols = (input[0].length - 1)*_stepSize + 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++){
            for (int j = 0; j < input[0].length; j++){
                output[i*_stepSize][j*_stepSize] = input[i][j];
                //streching the input by the step size
            }
        }
        return output;
    }
    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionForwardPass(input);
        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixInput);

    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, _inLength, _inRows, _inCols);
        backPropagation(matrixInput);

    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {


        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();
        //might have multiple input images which means filters may be applied more than once which means we will have a loss
        //for each filter
        //we need to find the derivative of the loss with respect to the filter
        //this will make it better for each input on average
        for( int f = 0; f < _filters.size(); f++){
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }
        for(int i = 0; i< _lastInput.size(); i++){

            double[][] errorForInput = new double[_inRows][_inCols];

            for (int f = 0; f< _filters.size(); f++){
                double[][] currentFilter = _filters.get(f);
                double[][] error = dLdO.get(i*_filters.size()+f);

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);

                //now apply our MatrixUtility class to update the filters
                double[][] delta = multiply(dLdF, _learningRate*-1);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = add(errorForInput,  fullConvolve(currentFilter, flippedError));
            }
            dLdOPreviousLayer.add(errorForInput);
        }
        //now we have the total delta for each filter
        for(int f = 0; f < _filters.size(); f++){
            double[][]  modified = add(filtersDelta.get(f), _filters.get(f));
            _filters.set(f, modified);
        }
        if (_previousLayer != null){
            _previousLayer.backPropagation(dLdOPreviousLayer);
        }

    }
    public double[][] flipArrayHorizontal(double[][] array){

        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                output[rows-i-1][j] = array[i][j];
            }
        }
        return output;
    }
    public double[][] flipArrayVertical(double[][] array){

        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                output[cols-i-1][j] = array[i][j];
            }
        }
        return output;
    }
    private double[][] fullConvolve(double[][] input, double[][] filter){
        int outRows = (input.length + filter.length)+ 1;
        int outCols = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];
        int outRow = 0;
        int outCol;
        //move filter across image

        for (int i = -fRows+1; i < inRows ; i++){
            outCol = 0;
            for (int j = -fCols; j < inCols ; j++){
                //apply filter to section of image
                double sum = 0.0;
                for (int x = 0; x < fRows; x++){
                    for (int y = 0; y < fCols; y++){
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols){
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum += value;
                        }

                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;

        }
        return output;
    }


    @Override
    public int getOutputLength() {
        return _filters.size()*_inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _filterSize)/_stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _filterSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength()*getOutputRows()*getOutputCols();
    }
}
