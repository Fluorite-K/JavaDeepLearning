package helper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Create by incognito on 2019-02-13
 */
public class FileIterator {
    /** 조건에 맞는 파일 리스트 */
    private final List<File> files;
    /** 현재까지 조회(생성)된 파일 해시 테이블 */
    private final HashMap<String, File> hashTable;

    /**
     * 생성자
     *
     * @param files 조회된 파일 객체 리스트
     * @param table 현재까지 생성된 파일 해시 테이블
     */
    private FileIterator(final List<File> files, final HashMap<String, File> table) {
        this.files = files;
        this.hashTable = table;
    }

    /**
     * FileIterator 기본 빌더 생성
     *
     * @return {@link Builder} 빌더 객체
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * FileIterator 순회를 시작할 루트 디렉터리를 포함하는 빌더 생성
     *
     * @param rootPath 순회를 시작할 루트 디렉터리 경로
     * @return {@link Builder} 빌더 객체
     */
    public static Builder builder(final String rootPath) {
        return new Builder().rootDir(rootPath);
    }

    /**
     * Map<String, File> -> Map<String, Path> 변환
     * @param fileMap
     * @return
     */
    static HashMap<String, Path> file2Path(final Map<String, File> fileMap) {
        final HashMap<String, Path> pathMap = new HashMap<>();
        fileMap.forEach((k, v) -> pathMap.put(k, Paths.get(k)));
        return pathMap;
    }

    /**
     * Map<String, Path> -> Map<String, File> 변환
     * @param pathMap
     * @return
     */
    static HashMap<String, File> path2File(final Map<String, Path> pathMap) {
        final HashMap<String, File> fileMap = new HashMap<>();
        pathMap.forEach((k, v) -> fileMap.put(k, v.toFile()));
        return fileMap;
    }

    /**
     * 순회 결과 {@link this#files}를 스트림 객체로 리턴.
     *
     * @return 파일 스트림 객체
     */
    public Stream<File> toStream() {
        return files.stream();
    }

    /**
     * 순회 결과 {@link this#files}를 리스트 객체로 리턴.
     *
     * @return 파일 리스트 객체
     */
    public List<File> toList() {
        return files;
    }

    /**
     * 순회 결과 {@link this#files} 개수 리턴.
     *
     * @return 조회된 파일 수
     */
    public int size() {
        return files.size();
    }

    /**
     * 조회(생성)된 파일 해시 테이블 리턴
     *
     * @return 조회(생성)된 파일 해시 테이블
     */
    public HashMap<String, File> getHashTable() {
        return hashTable;
    }


    /**
     * FileIterator를 생성하는 빌더
     *
     * [ FileIterator.builder(path) | builder.rootDir(path) ]
     * [ builder.excludePaths(paths) ] ?
     * [ builder.hashTable(map) ] ?
     * [ builder.isParallel(bool) ] ?
     * [ builder.sort(options) ] ?
     * [ builder.range(int...) | builder.rangeClosed(int...) ] ?
     *
     */
    public static class Builder {
        /** 순회를 시작할 루트 디렉터리 경로 */
        private String rootPath;
        /** 순회에서 배제할 디렉터리 혹은 파일 경로 */
        private final List<String> excludePaths = new ArrayList<>();
        /** 순회에서 배제할 패턴 */
        private final List<String> excludePatterns = new ArrayList<>();
        /** 순회 완료 혹은 기존에 순회했던 파일들의 해시 테이블. 테이블에 있는 파일은 조회하지 않음 */
        private final HashMap<String, Path> hashTable = new HashMap<>();
        /** 병렬 처리 여부: 기본 병렬처리 안 함. */
        private boolean isParallel = false;
        /** 파일 정렬 관련 옵션: 배열의 순서대로 적용됨 */
        private FileOrderOption[] sortingOptions;
        /** 전체 파일 리스트에서 slice 시작 인덱스 */
        private int startPosition;
        /** 전체 파일 리스트에서 slice 종료 인덱스 */
        private int endPosition;
        /** 전체 파일 리스트에서 현재 처리(조회) 중인 파일 위치(인덱스) */
        private AtomicInteger position = new AtomicInteger(0);
        /** 조회할 파일의 타입을 판단하는 함수: Default All Files */
        private FileType fileType = FileType.ALL;
        /** 처리 중인 파일 정보 출력할 logger: 기본 사용 안 함. */
        private Consumer<String> logger = str -> {};

        /** 기본 생성자 */
        private Builder() {}

        /**
         * 순회를 시작할 루트 파일 경로 지정
         * @param rootPath 순회 시작 파일 경로
         * @return
         */
        public Builder rootDir(final String rootPath) {
            this.rootPath = rootPath;
            return this;
        }

        /**
         * 조회할 파일의 타입 지정
         * @param fileType 파일 타입 [{@link FileType#ALL} | {@link FileType#DIRECTORY_ONLY} | {@link FileType#FILE_ONLY}]
         * @return
         */
        public Builder fileType(final FileType fileType) {
            this.fileType = fileType;
            return this;
        }

        /** 전체 파일에서 조회를 시작할 인덱스 지정
         *
         * @param startInclusive 시작 인덱스
         * @return
         */
        public Builder startInclusive(final int startInclusive) {
            this.startPosition = startInclusive;
            return this;
        }

        /**
         * 종료 인덱스 지정 (해당 인덱스 포함).
         * @{link this#rangeClosed(end)} 로 대체.
         *
         * @param endInclusive 종료 인덱스
         * @return
         */
        @Deprecated
        public Builder endInclusive(final int endInclusive) {
            this.endPosition = endInclusive;
            return this;
        }

        /**
         * 종료 인덱스 지정 (해당 인덱스 제외).
         * @{link this#range(end)} 로 대체.
         *
         * @param endExclusive 종료 인덱스
         * @return
         */
        @Deprecated
        public Builder endExclusive(final int endExclusive) {
            this.endPosition = endExclusive + 1;
            return this;
        }

        /**
         * 조회 시작, 종료 범위 지정
         * fileList[startInclusive : endExclusive - 1]
         *
         * @param startInclusive    시작 인덱스 (포함)
         * @param endExclusive      종료 인덱스 (제외)
         * @return
         */
        @SuppressWarnings("deprecation")
        public Builder range(final int startInclusive, final int endExclusive) {
            return this.startInclusive(startInclusive).endExclusive(endExclusive);
        }

        /**
         * 조회 종료 범위 지정
         * fileList[0 : endExclusive - 1]
         *
         * @param endExclusive      종료 인덱스 (제외)
         * @return
         */
        public Builder range(final int endExclusive) {
            return this.range(0, endExclusive);
        }

        /**
         * 조회 시작, 종료 범위 지정
         * fileList[startInclusive : endInclusive]
         *
         * @param startInclusive    시작 인덱스 (포함)
         * @param endInclusive      종료 인덱스 (포함)
         * @return
         */
        @SuppressWarnings("deprecation")
        public Builder rangeClosed(final int startInclusive, final int endInclusive) {
            return this.startInclusive(startInclusive).endInclusive(endInclusive);
        }

        /**
         * 조회 시작, 종료 범위 지정
         * fileList[0 : endInclusive]
         *
         * @param endInclusive      종료 인덱스 (포함)
         * @return
         */
        public Builder rangeClosed(final int endInclusive) {
            return rangeClosed(0, endInclusive);
        }

        /**
         * 순회에서 배제시킬 파일 목록 지정, 디렉터리 경로를 지정할 경우 해당 디렉터리는 순회 안 함.
         *
         * @param exclusives 제외할 리스트
         * @return
         */
        public Builder excludePaths(final String... exclusives) {
            this.excludePaths.addAll(Arrays.asList(exclusives));
            return this;
        }

        /**
         * 순회에서 배제할 경로의 정규표현식 패턴 지정
         * @param regex
         * @return
         */
        public Builder excludePatterns(final String... regex) {
            this.excludePatterns.addAll(Arrays.asList(regex));
            return this;
        }

        /**
         * 순회를 마친 파일 해시 테이블 지정: 테이블에 포함된 파일들은 조회 안 함.
         *
         * @param hashTable
         * @return
         */
        public Builder setHashTable(final Map<String, Path> hashTable) {
            this.hashTable.putAll(hashTable);
            return this;
        }

        /**
         * 순회를 마친 파일 경로 목록: 목록에 포함된 파일들은 조회 안 함.
         *
         * @param completes
         * @return
         */
        public Builder completedList(final Iterable<String> completes) {
            completes.forEach(path -> hashTable.put(path, Paths.get(path)));
            return this;
        }

        /**
         * 병렬 처리 지정
         *
         * @return
         */
        public Builder parallel() {
            this.isParallel = true;
            return this;
        }

        /**
         * 정렬 옵션 지정
         *
         * @param options
         * @return
         */
        public Builder sort(final FileOrderOption... options) {
            this.sortingOptions = options;
            return this;
        }

        /**
         * 커스텀 logger 지정
         * @param logger
         * @return
         */
        public Builder log(final Consumer<String> logger) {
            this.logger = logger;
            return this;
        }

        /**
         * 기본 logger 지정: System.out.println
         * @return
         */
        public Builder log() {
            return log(System.out::println);
        }

        /**
         * FileIterator 생성
         *
         * @return
         */
        public FileIterator build() {
            final Predicate<Path> chkDuplicate = checkDuplicatedFiles();
            try {
                final Stream<Path> paths = isParallel ? Files.walk(Paths.get(rootPath)).parallel() : Files.walk(Paths.get(rootPath)); // 병렬 처리 여부에 따라 스트림 생성
                final Stream<File> fileStream = paths
                            .filter(fileType.get()) // 지정된 파일 타입인지 판별
                            .filter(path -> excludePaths.stream().noneMatch(exclude -> path.toString().startsWith(exclude))) // 예외 목록 제외
                            .filter(chkDuplicate) // 해시 테이블에 존재하는 데이터 제외
                            .filter(path -> excludePatterns.stream().map(Pattern::compile).noneMatch(pattern -> pattern.matcher(path.toString()).matches())) // 정규표현식에 해당하는 파일 제외
                            .filter(path -> { // 지정된 인덱스 범위 이외의 데이터 제외
                                final int pos = position.getAndIncrement();
                                final boolean inRange = pos>= startPosition;
                                return endPosition > 0 ? inRange && pos <endPosition : inRange;
                            })
                            .map(String::valueOf) // Path -> String
                            .peek(path -> logger.accept(String.format("[#%d]\t%s", position.get(), path))) // logging
                            .map(File::new); // String -> File

                final Optional<Comparator<File>> comparator = maybeComparator();
                return new FileIterator(comparator.isPresent()
                        ? fileStream.sorted(comparator.get()).collect(Collectors.toList())
                        : fileStream.collect(Collectors.toList())
                    , path2File(hashTable));

            } catch (IOException e) {
                throw new InvalidFileException(e.getMessage(), e.getCause());
            }
        }

        /**
         * 정렬에 적용할 Comparator<File> 를 Optional로 래핑하여 리턴.
         * 정렬 옵션이 지정되지 않은 경우에는 Optional.empty() 리턴.
         * @return
         */
        private Optional<Comparator<File>> maybeComparator() {
            return Objects.isNull(sortingOptions) || sortingOptions.length < 1
                    ? Optional.empty()
                    : Stream.of(sortingOptions).map(FileOrderOption::get).reduce(Comparator::thenComparing);
        }

        /**
         * 해시 테이블에 존재하는지 여부 확인.
         * 해시 테이블에 존재하지 않는 새로운 파일의 경우 해시 테이블에 해당 파일 put.
         * put 할 때, 병렬 처리 여부에 따라 Race condition 발생할 수 있기에
         * 병렬 처리 여부와 해시 테이블의 데이터 존재 여부에 따라 {@link this::isNewFileSync} thread safe 메서드 수행
         *
         * @return
         */
        private Predicate<Path> checkDuplicatedFiles() {
            return !isParallel || hashTable.isEmpty() ? this::isNewFile : this::isNewFileSync;
        }

        /**
         * 해시 테이블에 존재하지 않는 새로운 파일인지 여부 확인.
         *
         * @param path 확인할 파일 경로 객체
         * @return 해시 테이블에 없음: true | 해시 테이블에 존재: false
         */
        private boolean isNewFile(final Path path) {
            final String key = path.toString();
            final boolean isNew = Objects.isNull(hashTable.get(key));
            if (isNew) {
                hashTable.put(key, path);
            }
            return isNew;
        }

        /**
         * [Thread safe] 해시 테이블에 존재하지 않는 새로운 파일인지 여부 확인.
         *
         * @param path 확인할 파일 경로 객체
         * @return 해시 테이블에 없음: true | 해시 테이블에 존재: false
         */
        private boolean isNewFileSync(final Path path) {
            final String key = path.toString();
            boolean isNew = Objects.isNull(hashTable.get(key));
            if (isNew) {
                synchronized (hashTable) {
                    isNew = isNewFile(path); // Double checked locking
                }
            }
            return isNew;
        }
    }

    /** 파일 관련 예외 Wrapping Class { checked exception -> runtime exception } */
    static class InvalidFileException extends RuntimeException {
        InvalidFileException(final String msg) { super(msg); }
        InvalidFileException(final Throwable t) { super(t); }
        InvalidFileException(final String msg, final Throwable t) { super(msg, t); }
    }

    /**
     * 파일 타입 판별을 위한 enum 클래스
     */
    public enum FileType {
        /** 해당 Path가 일반 파일을 참조할 경우 true */
        DIRECTORY_ONLY(Files::isDirectory),
        /** 해당 Path가 디렉터리를 참조할 경우 true */
        FILE_ONLY(Files::isRegularFile),
        /** 해당 Path가 파일이거나 디렉터리를 참조할 경우 true */
        ALL(path -> Files.isRegularFile(path) || Files.isDirectory(path));

        /** 파일 타입 구분 함수 */
        private Predicate<Path> predicate;
        /** 생성자 */
        FileType(final Predicate<Path> predicate) {
            this.predicate = predicate;
        }
        /** 파일 타입 판별 함수 리턴 */
        public Predicate<Path> get() {
            return predicate;
        }
    }

    /**
     * 파일 정렬 옵션 관련 enum 클래스
     */
    public enum FileOrderOption {
        /** 최근 수정일자 내림차순 */
        MOD_DATE_DESC(Comparator.comparing(File::lastModified).reversed()),
        /** 최근 수정일자 오름차순 */
        MOD_DATE_ASC(Comparator.comparing(File::lastModified)),
        /** 파일 경로 내림차순 */
        PATH_DESC(Comparator.comparing(File::getAbsolutePath).reversed()),
        /** 파일 경로 오름차순 */
        PATH_ASC(Comparator.comparing(File::getAbsolutePath));

        private Comparator<File> comparator;
        FileOrderOption(Comparator<File>comparator) { this.comparator = comparator; }
        public Comparator<File> get() { return comparator; }
    }
}