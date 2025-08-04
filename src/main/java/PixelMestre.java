import enums.Thresh;
import org.bytedeco.opencv.opencv_core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.*;

/**
 * Orquestra o fluxo de processamento, gerenciando a concorrência
 * e contendo a lógica de negócio.
 */
public class PixelMestre {

    private final ExecutorService executor;

    public PixelMestre() {
        int numThreads = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numThreads);
        System.out.println("PixelMestre iniciado com " + numThreads + " threads.");
    }

    public void executarBinarizacaoOtimizadaEmLote(String pastaEntrada, String pastaSaida) {
        long startTime = System.currentTimeMillis();
        List<Path> caminhosDasImagens = PixelCorreio.listarImagens(Paths.get(pastaEntrada));
        System.out.println(caminhosDasImagens.size() + " imagens encontradas para processamento.");

        for (Path caminhoDaImagem : caminhosDasImagens) {
            Callable<Void> tarefa = () -> {
                try {
                    Mat imagem = PixelCorreio.lerImagem(caminhoDaImagem.toString());

                    Mat imagemProcessada;
                    if (imagem.cols() > 2000) {
                        imagemProcessada = EcliPixel.binarizarParalelo(imagem, Thresh.OTSU);
                    } else {
                        imagemProcessada = EcliPixel.binarizar(imagem, Thresh.OTSU);
                    }

                    String nomeSaida = "processado_" + caminhoDaImagem.getFileName().toString();
                    Path caminhoFinal = Paths.get(pastaSaida).resolve(nomeSaida);
                    PixelCorreio.salvarImagem(caminhoFinal, imagemProcessada);

                } catch (Exception e) {
                    System.err.println("Erro ao processar " + caminhoDaImagem.getFileName() + ": " + e.getMessage());
                }
                return null;
            };
            executor.submit(tarefa);
        }

        desligar();
        long endTime = System.currentTimeMillis();
        System.out.println("Processamento em lote concluído em " + (endTime - startTime) / 1000.0 + " segundos.");
    }

    public void desligar() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
