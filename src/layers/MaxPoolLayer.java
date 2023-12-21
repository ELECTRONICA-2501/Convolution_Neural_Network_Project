package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {

    private int _stepSize;
    private int _windowSize;
    private int _inRows;
    private int _inCols;
    private int _inLength;

    List<int[][]> _lastMaxRow;
    //stores x coordinates of max values
    List<int[][]> _lastMaxCol;
    //stores y coordinates of max values

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inRows, int _inCols, int _inLength) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inRows = _inRows;
        this._inCols = _inCols;
        this._inLength = _inLength;
    }
    public List<double[][]> maxPoolForwardPass(List<double[][]> input){
        List<double[][]> output = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();

        for (int l = 0; l < input.size(); l++){
            output.add(pool(input.get(l)));
        }
        return output;

    }
    public double[][] pool(double[][] input){

        double[][] output = new double[getOutputRows()][getOutputCols()];
        int[][] maxRow = new int[getOutputRows()][getOutputCols()];
        int[][] maxCol = new int[getOutputRows()][getOutputCols()];

        for(int r = 0; r<getOutputRows(); r+= _stepSize){
            for(int c = 0; c<getOutputCols(); c+= _stepSize){

                double max = 0.0;
                maxRow[r][c] = -1;
                maxCol[r][c] = -1;
                //set to negative 1 so we know if there is a problem

                for (int x = 0; x < _windowSize; x++){
                    //x is the row
                    for (int y = 0; y < _windowSize; y++){
                        //y is the column
                        if (input[r+x][c+y] > max){
                            max = input[r+x][c+y];
                            //if the value is greater than the max, set the max to the value
                            maxRow[r][c] = r+x;
                            maxCol[r][c] = c+y;
                        }
                    }
                }
                output[r][c] = max;
            }
        }
        _lastMaxRow.add(maxRow);
        _lastMaxCol.add(maxCol);
        //now coordinates are being tracked
        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);

        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixList);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]> dXdL = new ArrayList<>();
        int l = 0;
        for(double[][] array: dLdO){
            double[][] error = new double[_inRows][_inCols];
            //this is the error matrix
            //find the value of error and set it to dldo
            for(int r = 0; r<getOutputRows(); r++){
                for (int c = 0; c<getOutputCols(); c++){
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if(max_i != -1){
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }
            dXdL.add(error);
            l++;
        }
        if(_previousLayer != null){
            _previousLayer.backPropagation(dXdL);
        }

    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _windowSize) / _stepSize + 1;
        //this is the formula for the output rows
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _windowSize) / _stepSize + 1;
        //this is the formula for the output columns
    }

    @Override
    public int getOutputElements() {
        return _inLength * getOutputRows() * getOutputCols();
    }
    //this class will have a 2d array to contain our data

}
