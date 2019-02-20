package cnn;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
//import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
//import org.datavec.image.recordreader.objdetect.impl.SvhnLabelProvider;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.SvhnLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import label_provider.PascalVocLabelProvider;
import lombok.extern.slf4j.Slf4j;

/**
 * Create by incognito on 2019-02-15
 */

/*
YOLO v2 Network

========================================================================================================================================================================
VertexName (VertexType)                       nIn,nOut    TotalParams   ParamsShape                                                  Vertex Inputs
========================================================================================================================================================================
input_1 (InputVertex)                         -,-         -             -                                                            -
conv2d_1 (ConvolutionLayer)                   3,32        864           W:{32,3,3,3}                                                 [input_1]
batch_normalization_1 (BatchNormalization)    32,32       128           gamma:{1,32}, beta:{1,32}, mean:{1,32}, var:{1,32}           [conv2d_1]
leaky_re_lu_1 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_1]
max_pooling2d_1 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_1]
conv2d_2 (ConvolutionLayer)                   32,64       18432         W:{64,32,3,3}                                                [max_pooling2d_1]
batch_normalization_2 (BatchNormalization)    64,64       256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, var:{1,64}           [conv2d_2]
leaky_re_lu_2 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_2]
max_pooling2d_2 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_2]
conv2d_3 (ConvolutionLayer)                   64,128      73728         W:{128,64,3,3}                                               [max_pooling2d_2]
batch_normalization_3 (BatchNormalization)    128,128     512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, var:{1,128}       [conv2d_3]
leaky_re_lu_3 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_3]
conv2d_4 (ConvolutionLayer)                   128,64      8192          W:{64,128,1,1}                                               [leaky_re_lu_3]
batch_normalization_4 (BatchNormalization)    64,64       256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, var:{1,64}           [conv2d_4]
leaky_re_lu_4 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_4]
conv2d_5 (ConvolutionLayer)                   64,128      73728         W:{128,64,3,3}                                               [leaky_re_lu_4]
batch_normalization_5 (BatchNormalization)    128,128     512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, var:{1,128}       [conv2d_5]
leaky_re_lu_5 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_5]
max_pooling2d_3 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_5]
conv2d_6 (ConvolutionLayer)                   128,256     294912        W:{256,128,3,3}                                              [max_pooling2d_3]
batch_normalization_6 (BatchNormalization)    256,256     1024          gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_6]
leaky_re_lu_6 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_6]
conv2d_7 (ConvolutionLayer)                   256,128     32768         W:{128,256,1,1}                                              [leaky_re_lu_6]
batch_normalization_7 (BatchNormalization)    128,128     512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, var:{1,128}       [conv2d_7]
leaky_re_lu_7 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_7]
conv2d_8 (ConvolutionLayer)                   128,256     294912        W:{256,128,3,3}                                              [leaky_re_lu_7]
batch_normalization_8 (BatchNormalization)    256,256     1024          gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_8]
leaky_re_lu_8 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_8]
max_pooling2d_4 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_8]
conv2d_9 (ConvolutionLayer)                   256,512     1179648       W:{512,256,3,3}                                              [max_pooling2d_4]
batch_normalization_9 (BatchNormalization)    512,512     2048          gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_9]
leaky_re_lu_9 (ActivationLayer)               -,-         0             -                                                            [batch_normalization_9]
conv2d_10 (ConvolutionLayer)                  512,256     131072        W:{256,512,1,1}                                              [leaky_re_lu_9]
batch_normalization_10 (BatchNormalization)   256,256     1024          gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_10]
leaky_re_lu_10 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_10]
conv2d_11 (ConvolutionLayer)                  256,512     1179648       W:{512,256,3,3}                                              [leaky_re_lu_10]
batch_normalization_11 (BatchNormalization)   512,512     2048          gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_11]
leaky_re_lu_11 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_11]
conv2d_12 (ConvolutionLayer)                  512,256     131072        W:{256,512,1,1}                                              [leaky_re_lu_11]
batch_normalization_12 (BatchNormalization)   256,256     1024          gamma:{1,256}, beta:{1,256}, mean:{1,256}, var:{1,256}       [conv2d_12]
leaky_re_lu_12 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_12]
conv2d_13 (ConvolutionLayer)                  256,512     1179648       W:{512,256,3,3}                                              [leaky_re_lu_12]
batch_normalization_13 (BatchNormalization)   512,512     2048          gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_13]
leaky_re_lu_13 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_13]
max_pooling2d_5 (SubsamplingLayer)            -,-         0             -                                                            [leaky_re_lu_13]
conv2d_21 (ConvolutionLayer)                  512,64      32768         W:{64,512,1,1}                                               [leaky_re_lu_13]
conv2d_14 (ConvolutionLayer)                  512,1024    4718592       W:{1024,512,3,3}                                             [max_pooling2d_5]
batch_normalization_21 (BatchNormalization)   64,64       256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, var:{1,64}           [conv2d_21]
batch_normalization_14 (BatchNormalization)   1024,1024   4096          gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_14]
leaky_re_lu_21 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_21]
leaky_re_lu_14 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_14]
space_to_depth_x2 (SpaceToDepth)              -,-         0             -                                                            [leaky_re_lu_21]
conv2d_15 (ConvolutionLayer)                  1024,512    524288        W:{512,1024,1,1}                                             [leaky_re_lu_14]
batch_normalization_15 (BatchNormalization)   512,512     2048          gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_15]
leaky_re_lu_15 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_15]
conv2d_16 (ConvolutionLayer)                  512,1024    4718592       W:{1024,512,3,3}                                             [leaky_re_lu_15]
batch_normalization_16 (BatchNormalization)   1024,1024   4096          gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_16]
leaky_re_lu_16 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_16]
conv2d_17 (ConvolutionLayer)                  1024,512    524288        W:{512,1024,1,1}                                             [leaky_re_lu_16]
batch_normalization_17 (BatchNormalization)   512,512     2048          gamma:{1,512}, beta:{1,512}, mean:{1,512}, var:{1,512}       [conv2d_17]
leaky_re_lu_17 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_17]
conv2d_18 (ConvolutionLayer)                  512,1024    4718592       W:{1024,512,3,3}                                             [leaky_re_lu_17]
batch_normalization_18 (BatchNormalization)   1024,1024   4096          gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_18]
leaky_re_lu_18 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_18]
conv2d_19 (ConvolutionLayer)                  1024,1024   9437184       W:{1024,1024,3,3}                                            [leaky_re_lu_18]
batch_normalization_19 (BatchNormalization)   1024,1024   4096          gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_19]
leaky_re_lu_19 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_19]
conv2d_20 (ConvolutionLayer)                  1024,1024   9437184       W:{1024,1024,3,3}                                            [leaky_re_lu_19]
batch_normalization_20 (BatchNormalization)   1024,1024   4096          gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_20]
leaky_re_lu_20 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_20]
concatenate_1 (MergeVertex)                   -,-         -             -                                                            [space_to_depth_x2, leaky_re_lu_20]
conv2d_22 (ConvolutionLayer)                  1280,1024   11796480      W:{1024,1280,3,3}                                            [concatenate_1]
batch_normalization_22 (BatchNormalization)   1024,1024   4096          gamma:{1,1024}, beta:{1,1024}, mean:{1,1024}, var:{1,1024}   [conv2d_22]
leaky_re_lu_22 (ActivationLayer)              -,-         0             -                                                            [batch_normalization_22]
conv2d_23 (ConvolutionLayer)                  1024,425    435625        W:{425,1024,1,1}, b:{1,425}                                  [leaky_re_lu_22]
outputs (Yolo2OutputLayer)                    -,-         0             -                                                            [conv2d_23]
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            Total Parameters:  50983561
        Trainable Parameters:  50983561
           Frozen Parameters:  0
========================================================================================================================================================================
 */
@Slf4j
public class YoloV2 {
    // parameters matching the pretrained YOLO2 model
    private static final long seed = 86347L;
    private static final int width = 416;
    private static final int height = 416;
    private static final int nChannels = 3;
    private static final int gridWidth = 13;
    private static final int gridHeight = 13;

    // number classes (digits) for dataset.
    private static final int nClasses = 8;

    // parameters for the Yolo2OutputLayer
    private static final double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
    private static final int nBoxes = 5;
    private static final double lambdaNoObj = 0.5;
    private static final double lambdaCoord = 1.0;
    private static final double detectionThreshold = 0.5;

    // parameters for the training phase
    private static final int batchSize = 2;
    private static final int nEpochs = 2;
    private static final double learningRate = 1e-4;
    private static final double lrMomentum = 0.9;

    private static void test() throws IOException {
        final Random rng = new Random(seed);
        SvhnDataFetcher fetcher = new SvhnDataFetcher();
        File trainDir = fetcher.getDataSetPath(DataSetType.TRAIN);
        File testDir = fetcher.getDataSetPath(DataSetType.TEST);


        log.info("Load data...");

        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new SvhnLabelProvider(trainDir));
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new SvhnLabelProvider(testDir));
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

//        IntStream.rangeClosed(1, 10).forEach(i -> {
//            System.err.println(test.next());
//        });
        for (long l : test.next().getFeatures().shape()) {
            System.err.print(l + " ");
        }

    }

    public static void main(String[] args) throws IOException {
        genYolo();
    }

    private static void genYolo() throws IOException {
        final Random rng = new Random(seed);

        final String removalLayer = "conv2d_23";

        log.info("Load data...");
        final String home = System.getProperty("user.home");
        final String trainImg = home+"/dev/dataset/VOCdevkit/VOC2007/train/";
        final String trainXml = home+"/dev/dataset/VOCdevkit/VOC2007/xml/train/";
        final String testImg = home+"/dev/dataset/VOCdevkit/VOC2007/test/";
        final String testXml = home+"/dev/dataset/VOCdevkit/VOC2007/xml/test/";

        FileSplit trainData = new FileSplit(new File(trainImg), NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testData = new FileSplit(new File(testImg), NativeImageLoader.ALLOWED_FORMATS, rng);

        System.out.println(testData.getRootDir());

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new PascalVocLabelProvider(new File(trainXml)));
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new PascalVocLabelProvider(new File(testXml)));
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        final ComputationGraph model;
        final String modelFilename = "yolo_v2_model.zip";

        if (new File(modelFilename).exists()) {
            log.info("Load model...");

            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Build model...");

            final ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained(PretrainedType.IMAGENET);
            final INDArray priors = Nd4j.create(priorBoxes);

            final FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                    .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();

            model = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)
//                    .removeVertexKeepConnections("conv2d_9")
                    .removeVertexKeepConnections(removalLayer)
                    .removeVertexKeepConnections("outputs")
                    .addLayer(removalLayer,
                            new ConvolutionLayer.Builder(1,1)
                                    .nIn(1024)
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1,1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_22")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambbaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            removalLayer)
                    .setOutputs("outputs")
                    .build();
            System.out.println(model.summary(InputType.convolutional(height, width, nChannels)));

            log.info("Train model...");

            model.setListeners(new ScoreIterationListener(1));
            for (int i = 0; i < nEpochs; i++) {
                train.reset();
                while (train.hasNext()) {
                    model.fit(train.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
        }
    }
}
