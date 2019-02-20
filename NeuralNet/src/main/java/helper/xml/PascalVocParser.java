package helper.xml;

import org.datavec.image.recordreader.objdetect.ImageObject;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import lombok.extern.slf4j.Slf4j;

/**
 * Create by incognito on 2019-02-19
 */
@Slf4j
public class PascalVocParser extends DefaultHandler {
    private static final List<String> BND_BOX_KEY = Arrays.asList("name", "xmin", "ymin", "xmax", "ymax");
    public static final String KEY_PATH = "filename";
//    public static final String KEY_PATH = "path";
    public static final String KEY_BOXES = "bndBoxes";

    private final SAXParser parser;

    private String xmlPath;
    private String filePath;
    private List<ImageObject> bndBoxes;

    private String currentTag;

    private BndBoxProvider provider;

    private final StringBuilder builder = new StringBuilder();

    public static Optional<Map<String, Object>> parse(final String xml) {
        try {
            final PascalVocParser instance = new PascalVocParser(xml);
            instance.parseXml();
            return Optional.of(instance.getResult());
        } catch (ParserConfigurationException | SAXException | IOException e) {
            log.error("SaxParser was not created.",  e);
        }
        return Optional.empty();
    }

    private PascalVocParser(final String xml) throws ParserConfigurationException, SAXException {
        xmlPath = xml;
        parser = SAXParserFactory.newInstance().newSAXParser();
        bndBoxes = new ArrayList<>();
    }

    private void parseXml() throws IOException, SAXException {
        parser.parse(Paths.get(xmlPath).toFile(), this);
    }

    @Override
    public void startElement(final String uri, final String localName, final String qName, final Attributes attributes) throws SAXException {
//        log.info("Current Tag: " + qName);
        currentTag = qName;
        if (qName.equals("object"))
            this.provider = BndBoxProvider.getWeakRef().get();
    }

    @Override
    public void characters(final char[] ch, final int start, final int length) throws SAXException {
        builder.setLength(0);
        builder.append(ch, start, length);
        final String text = builder.toString().trim();

        if (currentTag.equals(KEY_PATH)) {
            filePath = text;
        } else if (BND_BOX_KEY.contains(currentTag)) {
            provider.setValue(currentTag, text).ifPresent(bndBoxes::add);
        }
    }

    @Override
    public void endElement(String uri, String localName, String qName) throws SAXException {
        if (qName.equals("object"))
            this.provider = null;
        currentTag = "";
    }

    public Map<String, Object> getResult() {
        final Map<String, Object> res = new HashMap<>();
        res.put(KEY_PATH, filePath);
        res.put(KEY_BOXES, bndBoxes);
        return res;
    }

    private static class BndBoxProvider {
        private int[] points = new int[] {-1, -1, -1, -1};
        private String label;

        private BndBoxProvider() {}

        static WeakReference<BndBoxProvider> getWeakRef() { return new WeakReference<>(new BndBoxProvider()); }

        Optional<ImageObject> setValue(final String tag, final String value) {
            if (tag.equals("name")) return setLabel(value);
            return setPoint(tag, value);
        }

        private Optional<ImageObject> setPoint(final String tag, final String text) {
            final int pt = Integer.parseInt(text);
            switch (tag) {
                case "xmin":
                    points[0] = pt;
                    break;
                case "ymin":
                    points[1] = pt;
                    break;
                case "xmax":
                    points[2] = pt;
                    break;
                case "ymax":
                    points[3] = pt;
                    break;
            }
            return getBbox();
        }

        private Optional<ImageObject> setPosition(final int... pts) {
            this.points = pts;
            return getBbox();
        }

        private Optional<ImageObject> setLabel(final String label) {
            this.label = label;
            return getBbox();
        }

        private Optional<ImageObject> getBbox() {
            return isValidLabel() && isValidPosition()
                    ? Optional.of(new ImageObject(points[0], points[1], points[2], points[3], label))
                    : Optional.empty();
        }

        private boolean isValidPosition() {
            if (Objects.isNull(points)) return false;
            final List<Integer> pos = new ArrayList<>();
            for (int pt : points) {
                pos.add(pt);
            }
            return pos.stream().noneMatch(i -> i < 0);
        }

        private boolean isValidLabel() { return Objects.nonNull(label); }
    }
}
