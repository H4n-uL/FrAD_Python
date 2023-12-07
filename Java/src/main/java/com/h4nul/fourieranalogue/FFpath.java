package com.h4nul.fourieranalogue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public class FFpath {
    public static String ffmpeg;
    public static String ffprobe;

    static {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch");

        try {
            ffmpeg = extractExecutable(os, arch, "ffmpeg");
            ffprobe = extractExecutable(os, arch, "ffprobe");
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract FFmpeg executables", e);
        }
    }

    private static String extractExecutable(String os, String arch, String name) throws IOException {
        String resourceName = "/tools/" + name + "." + getExecutableSuffix(os, arch);
        InputStream resourceStream = FFpath.class.getResourceAsStream(resourceName);
        if (resourceStream == null) {
            throw new RuntimeException("Executable not found: " + resourceName);
        }

        Path tempFile = Files.createTempFile(name, getExecutableSuffix(os, arch));
        Files.copy(resourceStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
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

