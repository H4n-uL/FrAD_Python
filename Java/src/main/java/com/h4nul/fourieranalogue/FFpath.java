package com.h4nul.fourieranalogue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.util.HashSet;
import java.util.Set;

public class FFpath {
    public static String ffmpeg;
    public static String ffprobe;

    static {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch");

        try {
            ffmpeg = extractExecutable(os, arch, "ffmpeg");
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract FFmpeg executable", e);
        }
    }

    private static String extractExecutable(String os, String arch, String name) throws IOException {
        String resourceName = "/codec/" + name + "." + getExecutableSuffix(os, arch);
        InputStream resourceStream = FFpath.class.getResourceAsStream(resourceName);
        if (resourceStream == null) {
            throw new RuntimeException("Executable not found: " + resourceName);
        }

        Path tempFile = Files.createTempFile(name, getExecutableSuffix(os, arch));
        Files.copy(resourceStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
        Set<PosixFilePermission> perms = new HashSet<>();
        perms.add(PosixFilePermission.OWNER_EXECUTE);
        perms.add(PosixFilePermission.GROUP_EXECUTE);
        perms.add(PosixFilePermission.OTHERS_EXECUTE);
        Files.setPosixFilePermissions(tempFile, perms);

        tempFile.toFile().deleteOnExit();
        return tempFile.toAbsolutePath().toString();
    }

    private static String getExecutableSuffix(String os, String arch) {
        if (os.contains("windows")) {
            return "Windows";
        } else if (os.contains("mac")) {
            return "macOS";
        } else {
            if (arch.equals("amd64")) {
                return "AMD64";
            } else if (arch.equals("aarch64")) {
                return "AArch64";
            } else {
                throw new RuntimeException("Unsupported architecture: " + arch);
            }
        }
    }
}
