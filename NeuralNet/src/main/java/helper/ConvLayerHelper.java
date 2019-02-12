package helper;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.activations.Activation;

import java.util.stream.IntStream;

/**
 * Create by incognito on 2019-02-12
 */
public class ConvLayerHelper {

    public static ConvolutionLayer conv2d_3x3(final int inChannel, final int outChannel, final String padding) {
        final ConvolutionLayer.Builder conv = new ConvolutionLayer.Builder()
                .nIn(inChannel)
                .nOut(outChannel)
                .kernelSize(3, 3)
                .stride(1, 1)
                .activation(Activation.RELU);
//                .activation(Activation.LEAKYRELU);

        if (padding.toLowerCase().equals("same")) conv.padding(1, 1);

        return conv.build();
    }

    public static SubsamplingLayer max_pool_2x2() {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build();
    }

    public static NeuralNetConfiguration.ListBuilder conv2d_n_max_pood(final int ind, final int nConv, final NeuralNetConfiguration.ListBuilder builder, final int nChannel, final int nOutChannel, final String samePadding) {
        IntStream.rangeClosed(ind, ind + nConv).forEach(idx -> builder.layer(idx, conv2d_3x3(nChannel, nOutChannel, samePadding)));
        builder.layer((ind + nConv + 1), max_pool_2x2());
        return builder;
    }
}
