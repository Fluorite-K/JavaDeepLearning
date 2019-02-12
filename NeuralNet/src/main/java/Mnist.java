import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import lombok.extern.slf4j.Slf4j;

/**
 * Create by incognito on 2019-02-12
 */
@Slf4j
public class Mnist {
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int CHANNELS = 1; // single channel for grayscale images
    private static final int NUM_CLASSES = 10; // 10 digits classification
    private static final int BATCH_SIZE = 54;
    private static final int EPOCHS = 1;

    public static void main(String[] args) {
        final long seed = 46798L;
        final Random random = new Random(seed);

        final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        final DataSetIterator trainIter = dataSetIterator(new File("/Users/incognito/Downloads/mnist/mnist_png/training"), random);
        final DataSetIterator testIter = dataSetIterator(new File("/Users/incognito/Downloads/mnist/mnist_png/testing"), random);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        final MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd())
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
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
                        .nOut(NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
//                .backprop(true)
//                .pretrain(false)
                .build();

        final MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(10));
        log.debug("Total Parameter: {}", net.numParams());

        for (int i = 0; i < EPOCHS; i++) {
            net.fit(trainIter);
            log.debug("Completed epoch: {}", i);
            final Evaluation eval = net.evaluate(testIter);
            log.info("Stats {}", eval.stats());
            trainIter.reset();
            testIter.reset();
        }

        try {
            ModelSerializer.writeModel(net, new File("/Users/incognito/Downloads/mnist/mnist-model.zip"), true);
        } catch (IOException e) {
            log.error("Saving Error: " + e.getMessage(), e);
        }
    }

    private static DataSetIterator dataSetIterator(final File root, final Random random) {
        final FileSplit split = new FileSplit(root, NativeImageLoader.ALLOWED_FORMATS, random);
        final ImageRecordReader reader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, new ParentPathLabelGenerator());
        try {
            reader.initialize(split);
        } catch (IOException e) {
            log.error(e.getMessage(), e);
            return null;
        }
        return new RecordReaderDataSetIterator(reader, BATCH_SIZE, 1, NUM_CLASSES);
    }
}
