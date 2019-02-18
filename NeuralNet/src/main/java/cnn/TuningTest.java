package cnn;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.stream.IntStream;

import lombok.extern.slf4j.Slf4j;

/**
 * Create by incognito on 2019-02-15
 */
@Slf4j
public class TuningTest {
    protected static final int numClasses = 809;
    protected static final long seed = 12345;

    private static final String DATA_FILE = "data_cls.csv";
    private static final double trainPerc = 0.7; //The percent of train / test set ratio.
    private static final int batchSize = 64;
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
                                .activation(Activation.SOFTMAX).build(), "f2")
                .build();
        log.info(vgg16Transfer.summary());
//
//        TransferLearningHelper transferLearningHelper1 =
//                new TransferLearningHelper(pretrained, "fc2");
//        while(trainIter.hasNext()) {
//            DataSet currentFeaturized = transferLearningHelper1.featurize(trainIter.next());
//            saveToDisk(currentFeaturized, trainDataSaved, true);
//            trainDataSaved++;
//        }
//
//        TransferLearningHelper transferLearningHelper2 =
//                new TransferLearningHelper(vgg16Transfer);
//        while (trainIter.hasNext()) {
//            transferLearningHelper2.fitFeaturized(trainIter.next());
//        }

        //Dataset iterators
//        FlowerDataSetIterator.setup(batchSize,trainPerc);
//        DataSetIterator trainIter = FlowerDataSetIterator.trainIterator();
//        DataSetIterator testIter = FlowerDataSetIterator.testIterator();
//

        final RecordReader reader = new CSVRecordReader(',');
        reader.initialize(new FileSplit(new ClassPathResource("").getFile()));

        final int labelIndex = 0;
        final DataSetIterator iterator = new RecordReaderDataSetIterator(reader, batchSize, labelIndex, numClasses);
        final DataSet allData = iterator.next();
        allData.shuffle();
        final SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(trainPerc);  //Use 65% of data for training

        final DataSet trainingData = testAndTrain.getTrain();
        final DataSet testData = testAndTrain.getTest();

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        final DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set


        vgg16Transfer.init();
        vgg16Transfer.setListeners(new ScoreIterationListener(100));
        log.info("Model build complete");

        vgg16Transfer.fit(trainingData.iterateWithMiniBatches(), nEpochs);

        final Evaluation eval = new Evaluation(numClasses);
        final INDArray output = vgg16Transfer.outputSingle(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
    }
}
