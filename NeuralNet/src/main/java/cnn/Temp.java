package cnn;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Create by incognito on 19. 2. 18
 */
public class Temp {
    public static void main(String[] args) {
        final Path path = Paths.get(String.format("%s/%s", System.getProperty("user.home"), "Downloads/pokemon_jpg/train/abra/abra_2.jpg"));
        System.out.println(String.format("%s", path.toFile().exists() ));
    }
}
