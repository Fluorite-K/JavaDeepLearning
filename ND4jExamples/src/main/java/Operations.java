import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Create by incognito on 2019-02-12
 */
public class Operations {

    /**
     * Create Linear matrix values of start to end.
     *
     * @param startInclusive start value of matrix
     * @param endInclusive  end value of matrix
     * @param step  step value each elements
     * @return 2-D Array: {{start, [prev + step,]* end}}
     */
    public static INDArray arange(final double startInclusive, final double endInclusive, final double step) {
        final long elements = (long) ((endInclusive - startInclusive) / step); // count of elements except start
        return Nd4j.linspace(startInclusive, startInclusive + elements * step, elements+1);
    }

    /**
     * Create N-D array specified shape with random int values.
     * But each elements` actual type is double.
     *
     * @param shape N-D Array`s shape
     * @param upper maximum value of random int
     * @return
     */
    public static INDArray randInt(final int[] shape, final int upper) {
        return Transforms.floor(Nd4j.rand(shape).mul(upper));
    }

    public static String type(final INDArray array) {
        return array.data().dataType().name();
    }

    public static String arrayInfo(final INDArray array) {
        return array.data().dataType().toString();
    }

    public static INDArray reshape(final INDArray array, final int[] newShape) {
        final INDArray reshape = Nd4j.create(newShape);
        final long[] shape = array.shape();
        reshape.get(NDArrayIndex.createCoveringShape(shape)).assign(array);
        return reshape;
    }


    private static String getShapeString(final INDArray array) {
        final StringBuilder builder = new StringBuilder("[");
        final long[] shape = array.shape();
        for (int i = 0; i < shape.length; i++) {
            builder.append(shape[i]);
            if ((shape.length - 1) > i) builder.append(", ");
        }
        return builder.append("]").toString();
    }
    public static void main(String[] args) {
//        final List<INDArray> ndArrays = Arrays.asList(
//                arange(0, 10, 1)
//                , arange(0, 10, 2)
//                ,arange(0, 10, 3));
//        ndArrays.forEach(System.out::println);

//        System.out.println(randInt(new int[] {5, 4, 3}, 3));

        final INDArray array = randInt(new int[] {5, 4, 3}, 3);
//        System.out.println(type(array));
//        System.out.println(arrayInfo(array));
        System.out.println(reshape(array, new int[] {4, 5, 3}));
//        System.out.println(reshape(array, new int[] {10, 4, 4}));
    }
}
