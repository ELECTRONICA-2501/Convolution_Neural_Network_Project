package network;
import data.Image;
import layers.Layer;
import java.util.ArrayList;
import java.util.List;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class NeuralNetwork {

    //this class will represent the neural network
    //it will contain the layers of the network

    List<Layer> _layers;
    double scaleFactor;
    public NeuralNetwork(List<Layer> _layers, double scaleFactor){
        this._layers = _layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers(){
        if(_layers.size() <= 1){
            System.out.println("Not enough layers to link");
            return;
        }
        for(int i = 0; i < _layers.size() - 1; i++) {
            if(i ==0){
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            } else if (i == _layers.size() - 1){
                _layers.get(i).set_previousLayer(_layers.get(i-1));
            } else {
                _layers.get(i).set_previousLayer(_layers.get(i-1));
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer){
        int numClasses = networkOutput.length;
        //now create a vector of expected outputs
        double[] expectedOutput = new double[numClasses];
        expectedOutput[correctAnswer] = 1;

        return add(networkOutput, multiply(expectedOutput, -1));
    }

    private int getMaxIndex(double[] in){
        double max =0;
        int index = 0;
        for (int i = 0; i < in.length; i++){
            if (in[i] > max){
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    public int guess(Image image){
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(image.getData(), (1.0/scaleFactor)));

        double[] out = _layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        return guess;
    }

    public float test(List<Image> images){
        int correct = 0;
        for (Image image: images){
            int guess = guess(image);
            if (guess == image.getLabel()){
                correct++;
            }
        }
        return ((float)correct/images.size());
    }
    public void train (List<Image> images){

        for(Image image: images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(image.getData(), (1.0/scaleFactor)));

            double[] out = _layers.get(0).getOutput(inList);
            double[] dLdO = getErrors(out, image.getLabel());

            _layers.get((_layers.size()-1)).backPropagation(dLdO);
        }
    }
}
