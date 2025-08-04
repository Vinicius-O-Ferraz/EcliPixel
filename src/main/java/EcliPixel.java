import enums.CanalCor;
import enums.Thresh;
import exceptions.FalhaAoCarregarImagem;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static enums.Thresh.GLOBAL;
import static enums.Thresh.LOCAL_MEDIA;
import static org.bytedeco.opencv.global.opencv_core.split;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Classe de serviço stateless com métodos utilitários estáticos
 * para processamento de imagens usando OpenCV.
 */
public final class EcliPixel {

    // Construtor privado para impedir que a classe seja instanciada.
    private EcliPixel() {}

    // --- MÉTODOS DE PROCESSAMENTO DE IMAGEM ---

    public static Mat aplicarGaussian(Mat imagemEntrada, int kernelSize) {
        if (imagemEntrada.empty() || kernelSize <= 0 || kernelSize % 2 == 0) {
            throw new IllegalArgumentException("Imagem de entrada não pode ser vazia e o kernel deve ser um inteiro ímpar positivo.");
        }
        Mat imagemDesfocada = new Mat();
        GaussianBlur(imagemEntrada, imagemDesfocada, new Size(kernelSize, kernelSize), 0);
        return imagemDesfocada;
    }

    public static Mat converterCanalCores(Mat imagemEntrada, CanalCor canal) {
        if (imagemEntrada == null || imagemEntrada.empty()) {
            throw new IllegalArgumentException("A imagem de entrada não pode ser nula ou vazia.");
        }
        Mat imagemConvertida = new Mat();
        int escolha = switch (canal) {
            case HSV -> COLOR_BGR2HSV;
            case RGB -> COLOR_BGR2RGB;
            case GRAYSCALE -> COLOR_BGR2GRAY;
            case REVERSE -> COLOR_GRAY2BGR;
        };
        cvtColor(imagemEntrada, imagemConvertida, escolha);
        return imagemConvertida;
    }

    public static Mat binarizar(Mat imagemEntrada, Thresh metodo, Object... binparams) {
        if (imagemEntrada.empty()) {
            throw new IllegalArgumentException("A imagem de entrada para binarizar não pode ser vazia.");
        }
        Mat imagemBinaria = new Mat();
        Mat imagemCinza = (imagemEntrada.channels() == 3) ? converterCanalCores(imagemEntrada, CanalCor.GRAYSCALE) : imagemEntrada;

        switch (metodo) {
            case OTSU -> threshold(imagemCinza, imagemBinaria, 0, 255, THRESH_BINARY | THRESH_OTSU);
            case GLOBAL, GLOBAL_INV -> {
                if (binparams.length < 1) throw new IllegalArgumentException("Binarização GLOBAL requer 1 parâmetro: limiar (double).");
                double limiar = (double) binparams[0];
                int tipo = (metodo == GLOBAL) ? THRESH_BINARY : THRESH_BINARY_INV;
                threshold(imagemCinza, imagemBinaria, limiar, 255, tipo);
            }
            case LOCAL_MEDIA, LOCAL_GAUSSIANA -> {
                if (binparams.length < 2) throw new IllegalArgumentException("Binarização ADAPTATIVA requer 2 parâmetros: tamanhoBloco (int) e constanteC (double).");
                int tamanhoBloco = (int) binparams[0];
                double constanteC = (double) binparams[1];
                int tipo = (metodo == LOCAL_MEDIA) ? ADAPTIVE_THRESH_MEAN_C : ADAPTIVE_THRESH_GAUSSIAN_C;
                adaptiveThreshold(imagemCinza, imagemBinaria, 255, tipo, THRESH_BINARY, tamanhoBloco, constanteC);
            }
            default -> throw new UnsupportedOperationException("Tipo de binarização não implementado: " + metodo);
        }
        return imagemBinaria;
    }

    public static Mat isolarCanal(Mat imagemEntrada, int canalEscolhido) {
        if (imagemEntrada.empty() || imagemEntrada.channels() < 3) {
            throw new IllegalArgumentException("A imagem de entrada deve ser colorida (3 canais) para isolar um canal.");
        }
        if (canalEscolhido < 1 || canalEscolhido > 3) {
            throw new IllegalArgumentException("O canal escolhido deve ser 1, 2 ou 3.");
        }
        MatVector canais = new MatVector();
        split(imagemEntrada, canais);
        // Retorna uma cópia para evitar problemas de referência com o MatVector
        return new Mat(canais.get(canalEscolhido - 1));
    }

    public static Mat calcularHistograma(Mat imagemEntrada) {
        if (imagemEntrada.empty()) {
            throw new IllegalArgumentException("A imagem de entrada para o histograma não pode ser vazia.");
        }
        Mat imagemCinza = (imagemEntrada.channels() == 3) ? converterCanalCores(imagemEntrada, CanalCor.GRAYSCALE) : imagemEntrada;

        MatVector listaDeImagens = new MatVector(imagemCinza);
        Mat histograma = new Mat();
        Mat mascara = new Mat();
        IntPointer canais = new IntPointer(0);
        IntPointer tamanhoHist = new IntPointer(256);
        FloatPointer faixas = new FloatPointer(0f, 256f);

        calcHist(listaDeImagens, canais, mascara, histograma, tamanhoHist, faixas, true);
        return histograma;
    }

    public static Mat binarizarParalelo(Mat imagemEntrada, Thresh metodo, Object... binparams) {
        Mat imagemCinza = (imagemEntrada.channels() == 3) ? converterCanalCores(imagemEntrada, CanalCor.GRAYSCALE) : imagemEntrada;
        Mat imagemBinarizada = new Mat(imagemCinza.size(), imagemCinza.type());

        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        int alturaTotal = imagemCinza.rows();
        int alturaFatia = alturaTotal / numThreads;

        for (int i = 0; i < numThreads; i++) {
            int yInicial = i * alturaFatia;
            int alturaAtual = (i == numThreads - 1) ? (alturaTotal - yInicial) : alturaFatia;

            Rect regiao = new Rect(0, yInicial, imagemCinza.cols(), alturaAtual);
            Mat fatiaEntrada = new Mat(imagemCinza, regiao);
            Mat fatiaSaida = new Mat(imagemBinarizada, regiao);

            Runnable tarefa = () -> {
                Mat resultadoFatia = binarizar(fatiaEntrada, metodo, binparams);
                resultadoFatia.copyTo(fatiaSaida);
            };
            executor.submit(tarefa);
        }

        executor.shutdown();
        try {
            executor.awaitTermination(5, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return imagemBinarizada;
    }
}
