package org.deeplearning4j.examples.dataexamples;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.dataexamples.MnistImagePipelineExample;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/* Create model using Yann LeCun's MNIST dataset. Train model and save as zip.
 * Followed https://github.com/deeplearning4j/dl4j-examples
 */
public class CreateModelTrainSave {
    private static Logger log = LoggerFactory.getLogger(CreateModelTrainSave.class);

    /**
     * Data URL for downloading
     */
    public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    /**
     * Location to save and extract the training/testing data
     */
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

    public static void main(String[] args) throws Exception {

        int height = 28;
        int width = 28;
        int channels = 1; // Grayscale.
        int seed = 123;
        Random randNumGen = new Random(seed); // Reproducible initial weights.
        int batchSize = 64; // How many examples to fetch with each step. Faster training.
        int outputNum = 10; // Number of possible outcomes.
        int numEpochs = 1; // An epoch is a complete pass through a given dataset. Result in better accuracy.
        int iterations = 1; // How many complete batch cycle to update weights.

        MnistImagePipelineExample.downloadData();
        // Define the File Paths
        File trainData = new File(DATA_PATH + "/mnist_png/training");
        File testData = new File(DATA_PATH + "/mnist_png/testing");

        // Define the FileSplit(PATH, ALLOWED FORMATS, random)
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        //Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Reads a local file system and parses images of a given dimensions with labels
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // Initialize the record reader
        recordReader.initialize(train);

        // DataSet Iterator used to fetch the dataset.
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // Build Neural Network
        log.info("BUILDING MODEL");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // Keeping same initialization allowing us to tinker model.
                .iterations(iterations)
                .regularization(true).l2(0.0005) // Prevents overfitting by preventing  individual weights from having too much influence on the overall results.
                .learningRate(.01) // Size of adjustment made to the weights each iteration.
                .weightInit(WeightInit.XAVIER) // As per Glorot and Bengio 2010: Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // Loss optimalization algorithm.
                .updater(Updater.NESTEROVS) // Gives velocity to optimalization algorithm.
                .list() // Replicates configuration n times and builds a layerwise configuration.
                .layer(0, new ConvolutionLayer.Builder(5, 5) // 5x5 filter that convolutes over input
                        .nIn(channels)
                        .stride(1, 1) // Step of filter
                        .nOut(20)
                        .activation(Activation.IDENTITY) // Activation function
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // Pooling layer type MAX
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum) // Number of output classes
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        // output to show how well the network is training
        model.setListeners(new ScoreIterationListener(50));

        log.info("************TRAINING MODEL**************");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(dataIter); // train model using data
            log.info("*** Completed epoch {} ***", i);
        }

        log.info("**************EVALUATE MODEL**************");

        // Evaluate model using test data
        recordReader.reset();
        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        log.info(recordReader.getLabels().toString());

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(outputNum);

        // Evaluate the network
        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            eval.eval(next.getLabels(), output);
        }

        log.info(eval.stats());


        log.info("*****************SAVE TRAINED MODEL*******************");
        // Where to save model
        File locationToSave = new File(DATA_PATH + "trained_mnist_model.zip");

        // boolean save Updater. True for further training.
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        log.info("****************Example finished********************");
    }
}
