import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;

import static helper.ConvLayerHelper.*;

/**
 * Create by incognito on 2019-02-12
 */
public class SingleShotMultiBoxDetector {
    public static final long SEED = 26470;

    public static MultiLayerConfiguration getConfig(final long[] inputShape, final long[] outputShape, final int batchSize, final int epoch){
        final int nChannel = 3;
        final NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
            .seed(SEED)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(0.01, 0.9))
            .weightInit(WeightInit.XAVIER)
            .biasInit(0)
            .l2(0.0005)
            .list();

        // Add Layer (conv2d * 2) + max_pool : 0 ~ 2
        conv2d_n_max_pood(0, 2, builder, 3, 64, "same");
        // Add Layer (conv2d * 2) + max_pool : 3 ~ 5
        conv2d_n_max_pood(3, 2, builder, 64, 128, "same");
        // Add Layer (conv2d * 3) + max_pool : 6 ~ 9
        conv2d_n_max_pood(6, 3, builder, 128, 256, "same");
        // Add Layer (conv2d * 3) + max_pool : 10 ~ 13
        conv2d_n_max_pood(10, 3, builder, 256, 512, "same");

        builder.layer(14, conv2d_3x3(512, 512, "same")) // conv_5_1
                .layer(15, conv2d_3x3(512, 512, "same")) // conv_5_2
                .layer(16, conv2d_3x3(512, 512, "same")) // conv_5_3
                .layer(17, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // pool_5
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .build())
                .layer(18, new ConvolutionLayer.Builder()   // fc6
                        .nIn(512)
                        .nOut(1024)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(6, 6)
                        .dilation(6, 6)
                        .activation(Activation.RELU).build())
                .layer(19, new ConvolutionLayer.Builder()   // fc7
                        .nIn(1024)
                        .nOut(1024)
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .activation(Activation.RELU).build())
                .layer(20, new ConvolutionLayer.Builder()   // conv_6_1
                        .nIn(1024)
                        .nOut(256)
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(21, new ConvolutionLayer.Builder()   // conv_6_2
                        .nIn(256)
                        .nOut(512)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(22, new ConvolutionLayer.Builder()   // conv_7_1
                        .nIn(512)
                        .nOut(128)
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(23, new ConvolutionLayer.Builder()   // conv_7_2
                        .nIn(128)
                        .nOut(256)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(24, new ConvolutionLayer.Builder()   // conv_8_1
                        .nIn(256)
                        .nOut(128)
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(25, new ConvolutionLayer.Builder()   // conv_8_2
                        .nIn(128)
                        .nOut(256)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(26, new ConvolutionLayer.Builder()   // conv_9_1
                        .nIn(256)
                        .nOut(128)
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(27, new ConvolutionLayer.Builder()   // conv_9_2
                        .nIn(128)
                        .nOut(256)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build());

        return builder.build();
    }
}
