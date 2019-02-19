package label_provider;

import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;

import java.io.File;
import java.net.URI;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import helper.file_system.FileIterator;
import helper.xml.PascalVocParser;

/**
 * Create by incognito on 2019-02-19
 */
public class PascalVocLabelProvider implements ImageObjectLabelProvider {
    private final Map<String, List<ImageObject>> labelMap;

    public PascalVocLabelProvider(final File root) {
        labelMap = new HashMap<>();

        final Pattern pattern = Pattern.compile("(.+?)\\.xml");
        FileIterator.builder(root.getAbsolutePath())
                .parallel()
                .fileType(FileIterator.FileType.FILE_ONLY)
                .build().toStream()
                .parallel()
                .filter(file -> pattern.matcher(file.getName()).matches())
                .map(File::getAbsolutePath)
                .forEach(path ->
                    PascalVocParser.parse(path).ifPresent(result ->
                        labelMap.put(String.valueOf(result.get(PascalVocParser.KEY_PATH)), (List<ImageObject>) result.get(PascalVocParser.KEY_BOXES))
                    )
                );

        System.out.println(labelMap);
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(final String path) {
        return labelMap.get(path);
    }

    @Override
    public List<ImageObject> getImageObjectsForPath(final URI uri) {
        return getImageObjectsForPath(uri.toString());
    }

    public static void main(String[] args) {
        new PascalVocLabelProvider(Paths.get("/Users/incognito/Downloads/backup/python/workspace/keras-yolo3/VOCdevkit/VOC2007/Annotations").toFile());
    }
}
