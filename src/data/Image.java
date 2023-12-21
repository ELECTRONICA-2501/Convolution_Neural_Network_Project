package data;

public class Image {
    //this class will have a 2d array to contain our data
    //no need for setters since we dont wanna change our labels
    private double[][] data;

    private int label;

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }
    public int getLabel() {
        return label;
    }
    public double[][] getData() {
        return data;
    }
    @Override
    public String toString(){
        //this method will return a string representation of the image
        String s = label + ", \n";

        for (int i = 0; i < data.length; i++){
            for (int j = 0; j< data[0].length; j++){
                s += data[i][j] + ", ";
            }
            s += "\n";
        }
        return s;
    }
}
