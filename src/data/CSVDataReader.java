package data;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;
import java.util.ArrayList;

public class CSVDataReader {
    //this class is responsible for reading the data from the MNIST csv files

    private final int ROWS = 28;
    private final int COLS = 28;
    //

    public List<Image> readData(String path){
        //method to read the data from the csv file
        List<Image> images = new ArrayList<>();

        try(BufferedReader dataReader = new BufferedReader(new FileReader(path))){
            String line;

            while((line = dataReader.readLine()) != null){
                //while we are still getting data
                String[] lineItems = line.split(",");
                double[][] data = new double[ROWS][COLS];
                int label = Integer.parseInt(lineItems[0]);
                //gets the first number in the csvfile

                int index = 1;
                for (int row = 0; row < ROWS; row++){
                    for (int col = 0; col < COLS; col++){
                        //loop through the rows and columns
                        data[row][col] = (double) Integer.parseInt(lineItems[index++]);
                        //add the data to the 2d array
                    }
                }
                images.add(new Image(data, label));
            }


        }catch (Exception e){
            e.printStackTrace();
        }
        return images;
//
    }
}
