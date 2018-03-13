package edu.nyamdorj;

public class App 
{
    public static void main( String[] args ) throws Exception {


        int result = LoadModelWithFileChooser.getPredictionWithFileChooser();
        int result2 = LoadModel.getPrediction("/home/nyam/0.png");
        System.out.println("Prediction of chosen image: "+result);
        System.out.println("I passed image of 0 grayscale. Prediction was: "+result2);

    }
}
