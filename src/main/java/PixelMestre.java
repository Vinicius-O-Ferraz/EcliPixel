import org.bytedeco.opencv.opencv_core.Mat;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.Function;

public class PixelMestre {

    private final ExecutorService executor;

    public PixelMestre() {
        int numThreads = Runtime.getRuntime().availableProcessors();
        this.executor = Executors.newFixedThreadPool(numThreads);
        System.out.println("PixelMestre iniciado com " + numThreads + " threads.");
    }

    public void executarEmLote(String pastaEntrada, String pastaSaida, Function<Mat, Mat> pipelineDeProcessamento) {
        long startTime = System.currentTimeMillis();
        List<Path> caminhosDasImagens = PixelCorreio.listarImagens(Paths.get(pastaEntrada));
        System.out.println("\nIniciando lote: " + caminhosDasImagens.size() + " imagens encontradas.");

        for (Path caminhoDaImagem : caminhosDasImagens) {
            Callable<Void> tarefa = () -> {
                try {
                    Mat imagem = PixelCorreio.lerImagem(caminhoDaImagem.toString());

                    // Aplica o pipeline de processamento fornecido
                    Mat imagemProcessada = pipelineDeProcessamento.apply(imagem);

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
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) { executor.shutdownNow(); }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
