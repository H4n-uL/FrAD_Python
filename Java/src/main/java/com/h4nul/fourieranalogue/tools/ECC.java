// package com.h4nul.fourieranalogue.tools;

// import com.backblaze.erasure.ReedSolomon;
// import java.util.concurrent.*;

// public class ECC {
//     public static byte[][] splitData(byte[] data, int chunkSize) {
//         int numChunks = data.length / chunkSize;
//         if (data.length % chunkSize != 0) {
//             numChunks++;
//         }
//         byte[][] chunks = new byte[numChunks][chunkSize];
//         for (int i = 0; i < numChunks; i++) {
//             System.arraycopy(data, i * chunkSize, chunks[i], 0, chunkSize);
//         }
//         return chunks;
//     }

//     public static class Rdsl {
//         public static ReedSolomon rs = ReedSolomon.create(20, 148);

//         public static byte[] encodeChunk(byte[] chunk) {
//             byte[][] shards = splitData(chunk, 128);
//             rs.encodeParity(shards, 0, shards.length);
//             return concatenate(shards);
//         }

//         public static byte[] decodeChunk(byte[] chunk) {
//             byte[][] shards = splitData(chunk, 148);
//             try {
//                 rs.decodeMissing(shards, new boolean[shards.length], 0, shards[0].length);
//             } catch (Exception e) {
//                 System.out.println("Error: " + e);
//                 return null;
//             }
//             return concatenate(shards);
//         }

//         public static byte[] encode(byte[] data) {
//             byte[][] chunks = splitData(data, 128);
//             ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() / 2);
//             List<Future<byte[]>> futures = new ArrayList<>();
//             for (byte[] chunk : chunks) {
//                 futures.add(executor.submit(() -> encodeChunk(chunk)));
//             }
//             executor.shutdown();
//             byte[][] encodedChunks = new byte[futures.size()][];
//             for (int i = 0; i < futures.size(); i++) {
//                 try {
//                     encodedChunks[i] = futures.get(i).get();
//                 } catch (InterruptedException | ExecutionException e) {
//                     e.printStackTrace();
//                 }
//             }
//             return concatenate(encodedChunks);
//         }

//         public static byte[] decode(byte[] data) {
//             byte[][] chunks = splitData(data, 148);
//             ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() / 2);
//             List<Future<byte[]>> futures = new ArrayList<>();
//             for (byte[] chunk : chunks) {
//                 futures.add(executor.submit(() -> decodeChunk(chunk)));
//             }
//             executor.shutdown();
//             byte[][] decodedChunks = new byte[futures.size()][];
//             for (int i = 0; i < futures.size(); i++) {
//                 try {
//                     decodedChunks[i] = futures.get(i).get();
//                 } catch (InterruptedException | ExecutionException e) {
//                     e.printStackTrace();
//                 }
//             }
//             return concatenate(decodedChunks);
//         }
//     }

//     public static byte[] encode(byte[] data, boolean isEccOn) {
//         if (isEccOn) {
//             return Rdsl.encode(data);
//         }
//         return data;
//     }

//     public static byte[] decode(byte[] data) {
//         return Rdsl.decode(data);
//     }
// }
