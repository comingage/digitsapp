package edu.nyamdorj;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;

/* User provide image using file chooser dialog. Load model. Scale input. Get prediction.
 *
 */
public class LoadModelWithFileChooser {
    private static Logger log = LoggerFactory.getLogger(LoadModelWithFileChooser.class);

    /**
     * Location to save and extract the training/testing data
     */
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");


    private static String fileChose() {
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            return file.getAbsolutePath();
        } else {
            return null;
        }
    }


    public static int getPredictionWithFileChooser() throws Exception {
        int height = 28;
        int width = 28;
        int channels = 1;

        List<Integer> labelList = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

        // pop up file chooser
        String filechose = fileChose();

        //LOAD NEURAL NETWORK

        // Where to save model
        File locationToSave = new File(DATA_PATH + "trained_mnist_model.zip");
        // Check for presence of saved model
        if (locationToSave.exists()) {
            log.info("Saved Model Found!");
        } else {
            log.error("File not found!");
            log.error("This example depends on running CreateModelTrainSave, run that example first");
            System.exit(0);
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        // FileChose is a string we will need a file
        File file = new File(filechose);

        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray
        INDArray image = loader.asMatrix(file);

        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        // Pass through to neural Net
        INDArray output = model.output(image);

        log.info("The file chosen was " + filechose);
        log.info("The neural nets prediction (list of probabilities per label)");
        log.info("## List of Labels in Order## ");
        log.info(output.toString());
        log.info(labelList.toString());
        log.info("Getting predicted value as int");

        INDArray predictionIndex = output.argMax();
        return predictionIndex.getInt(0, 0);

    }

    public static void main(String[] args) throws Exception {
        getPredictionWithFileChooser();
    }
}
