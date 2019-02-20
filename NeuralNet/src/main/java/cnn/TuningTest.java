package cnn;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.meta.Prediction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import lombok.extern.slf4j.Slf4j;

/**
 * Create by incognito on 2019-02-15
 */
@Slf4j
public class TuningTest {
    protected static final int height = 224;
    protected static final int width = 224;
    protected static final int channel = 3;
    protected static final int numClasses = 809;
    protected static final long seed = 12345;

    private static final String HOMEDIR = System.getProperty("user.home");
    private static final Path TRAIN_DATA_FILE = Paths.get(HOMEDIR + "/Downloads/pokemon_jpg/train/");
    private static final Path TEST_DATA_FILE = Paths.get(HOMEDIR + "/Downloads/pokemon_jpg/test/");
    private static final double trainPerc = 0.7; //The percent of train / test set ratio.
    private static final int batchSize = 5;
    private static final String featureExtractionLayer = "fc2";

    private static final int nEpochs = 100;

    public static void main(String[] args) throws IOException, InterruptedException {
        final VGG16 model = VGG16.builder().build();
        final ComputationGraph pretrained = (ComputationGraph) model.initPretrained(PretrainedType.IMAGENET);
        log.info(pretrained.summary());

        final FineTuneConfiguration conf = new FineTuneConfiguration.Builder()
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();

        final ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(conf)
                .setFeatureExtractor(featureExtractionLayer)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numClasses)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0, 0.2*(2.0 / (4096+numClasses))))
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();
        log.info(vgg16Transfer.summary());


        log.info("Load Dataset...");

        final Random randNumGen = new Random(seed);

        final File trainData = TRAIN_DATA_FILE.toFile();
        final File testData = TEST_DATA_FILE.toFile();
        final FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        final FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // Extract the parent path as the image label
        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        final ImageRecordReader recordReader = new ImageRecordReader(height, width, channel, labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name
        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator
        final DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        final DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);

        // Scale pixel values to 0-1
        final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);// new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);


        vgg16Transfer.init();
        vgg16Transfer.setListeners(new ScoreIterationListener(100));
        log.info("Model build complete");

        for (int i = 0; i < 1; i++) {
            vgg16Transfer.fit(trainIter);
        }
        log.info("Training done");

//
//        for( int i=0; i<50; i++ ) {
//            vgg16Transfer.fit(trainingData);
//        }
//
//        //Evaluate the model on the test set
//        Evaluation eval = new Evaluation(3);
//        INDArray output = vgg16Transfer.outputSingle(testData.getFeatures());
//        eval.eval(testData.getLabels(), output, testMetaData);          //Note we are passing in the test set metadata here
//        System.out.println(eval.stats());
//
//        //Get a list of prediction errors, from the Evaluation object
//        //Prediction errors like this are only available after calling iterator.setCollectMetaData(true)
//        List<Prediction> predictionErrors = eval.getPredictionErrors();
//        System.out.println("\n\n+++++ Prediction Errors +++++");
//        for(Prediction p : predictionErrors){
//            System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
//                    + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
//        }
//
//        //We can also load a subset of the data, to a DataSet object:
//        List<RecordMetaData> predictionErrorMetaData = new ArrayList<>();
//        for( Prediction p : predictionErrors ) predictionErrorMetaData.add(p.getRecordMetaData(RecordMetaData.class));
//        DataSet predictionErrorExamples = iterator.loadFromMetaData(predictionErrorMetaData);
//        normalizer.transform(predictionErrorExamples);  //Apply normalization to this subset
//
//        //We can also load the raw data:
//        List<Record> predictionErrorRawData = reader.loadFromMetaData(predictionErrorMetaData);
//
//        //Print out the prediction errors, along with the raw data, normalized data, labels and network predictions:
//        for(int i=0; i<predictionErrors.size(); i++ ){
//            Prediction p = predictionErrors.get(i);
//            RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
//            INDArray features = predictionErrorExamples.getFeatures().getRow(i);
//            INDArray labels = predictionErrorExamples.getLabels().getRow(i);
//            List<Writable> rawData = predictionErrorRawData.get(i).getRecord();
//
//            INDArray networkPrediction = vgg16Transfer.outputSingle(features);
//
//            System.out.println(meta.getLocation() + ": "
//                    + "\tRaw Data: " + rawData
//                    + "\tNormalized: " + features
//                    + "\tLabels: " + labels
//                    + "\tPredictions: " + networkPrediction);
//        }
//
//
//        //Some other useful evaluation methods:
//        List<Prediction> list1 = eval.getPredictions(1,2);                  //Predictions: actual class 1, predicted class 2
//        List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     //All predictions for predicted class 2
//        List<Prediction> list3 = eval.getPredictionsByActualClass(2); //All predictions for actual class 2
    }
}
