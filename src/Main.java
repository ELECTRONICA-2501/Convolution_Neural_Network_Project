import data.CSVDataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;
import java.util.Random;

import java.util.List;

import static java.util.Collections.shuffle;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    public static void main(String[] args) {

        long SEED = 123;

        System.out.println("Hello World!, Starting data loading...");

        List<Image> imagesTest = new CSVDataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new CSVDataReader().readData("data/mnist_train.csv");

        //System.out.printf(images.get(0).toString());
        System.out.println("Image Train Size: " + imagesTrain.size());
        System.out.println("Image Test Size: " + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        NeuralNetwork net = builder.build();

        float rate = net.test(imagesTest);
        System.out.println("Pre trainging success rate: " + rate);

        int epochs = 3;

        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + i);
            shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Post training success rate: " +i+ ": " + rate);
        }

    }
}
