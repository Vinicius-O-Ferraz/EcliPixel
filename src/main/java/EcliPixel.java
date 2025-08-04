import enums.CanalCor;
import enums.Thresh;
import exceptions.FalhaAoCarregarImagem;
import lombok.Data;
import static enums.Thresh.GLOBAL;
import static enums.Thresh.LOCAL_MEDIA;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Size;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.bytedeco.opencv.global.opencv_core.split;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;


@Data
public class EcliPixel {
    //essa classe vai conter os métodos de processamento de imagem em si
    //funciona como um 'serviço' acessível de outros lugares do código
    private Path outPath;
    private final Mat imagemPura;

    public EcliPixel(String input, String output, Boolean resource){
        this.imagemPura= resource ? lerImagemDeRecurso(input) : lerImagem(input);
        this.outPath= Paths.get(output);
    }
    public EcliPixel(Mat imagemPura, String output){
        this.imagemPura= imagemPura;
        this.outPath= Paths.get(output);
    }

    // ler de um recurso do classpath
    public static Mat lerImagemDeRecurso(String caminhoNoRecurso) {
        try {
            URL url = EcliPixel.class.getClassLoader().getResource(caminhoNoRecurso);
            if (url == null) {
                throw new FalhaAoCarregarImagem("Recurso não encontrado no classpath: " + caminhoNoRecurso);
            }
            Path pathDoRecurso = Paths.get(url.toURI());
            Mat imagem = imread(pathDoRecurso.toString());
            if (imagem.empty()) {
                throw new FalhaAoCarregarImagem("Falha ao carregar a imagem do recurso: " + pathDoRecurso);
            } else {
                System.out.println("Imagem lida: " + pathDoRecurso.getFileName());
                return imagem;
            }
        } catch (Exception e) {
            throw new RuntimeException("Erro ao processar o caminho do recurso", e);
        }
    }

    public static Mat lerImagem(String path){
        Mat imagem = imread(path, IMREAD_COLOR);
        if (imagem.empty()){
            throw new FalhaAoCarregarImagem("falha ao carregar a imagem contida em:\n" + path);
        }
        else {
            System.out.println("Imagem: " + imagem.cols() + imagem.rows());
            return imagem;
        }
    }

    public void salvarImagem(String nomeDoArquivo, Mat imagemParaSalvar) {
        if (this.outPath == null) {
            throw new IllegalStateException("O caminho de saída (outPath) não foi definido.");
        }
        try {
            Files.createDirectories(this.outPath); // Cria a pasta 'output' se não existir
            Path caminhoFinal = this.outPath.resolve(nomeDoArquivo);
            imwrite(caminhoFinal.toString(), imagemParaSalvar);
            System.out.println("Imagem salva em: " + caminhoFinal.toAbsolutePath());
        } catch (IOException e) {
            throw new RuntimeException("Falha ao criar diretórios para o caminho de saída.", e);
        }
    }

    public Mat aplicarGaussian(Integer kernelSize){
        Mat imagem = this.getImagemPura();
        Size kernel = new Size(kernelSize, kernelSize);
        if (imagem.empty() || kernelSize % 2 == 0 ){
            String message = imagem.empty() ? "a imagem carregada pode não existir" : "o kernel precisa ter tamanho ímpar!";
            throw new FalhaAoCarregarImagem(message);
        }
        Mat imagemDesfocada = new Mat();
        GaussianBlur(imagem, imagemDesfocada, kernel, 0);
        return imagemDesfocada;
    }

    public Mat converterCanalCores(Mat imagemOrig, CanalCor canal){
        Mat imagemConvertida= new Mat();
        if (imagemOrig == null || imagemOrig.empty()) {
            System.err.println("A imagem de entrada está vazia.");
            return null;
        }
        int escolha= switch (canal){
            case HSV -> COLOR_BGR2HSV;
            case RGB -> COLOR_BGR2RGB;
            case GRAYSCALE -> COLOR_BGR2GRAY;
            case REVERSE -> COLOR_GRAY2BGR;
        };
        cvtColor(imagemOrig, imagemConvertida, escolha);
        return imagemConvertida;
    }

    public Mat binarizar(Thresh metodo, Object... binparams){
        double valorMax= 255;
        Mat imagemBinaria= new Mat();
        Mat imagemCinza = new Mat();
        Mat imagem = this.getImagemPura();
        cvtColor(imagem, imagemCinza, COLOR_BGR2GRAY);

        switch (metodo){
            case OTSU:
                threshold(imagemCinza, imagemBinaria, 0, valorMax, THRESH_OTSU);
                break;
            case GLOBAL:
            case GLOBAL_INV:
                if (binparams.length < 1){
                    throw new IllegalArgumentException("Binarização SIMPLES requer 1 parâmetro: limiar (double).");
                }
                double limiar = (double) binparams[0];
                int tipoThresholdSimples = (metodo == GLOBAL) ? THRESH_BINARY : THRESH_BINARY_INV;
                threshold(imagemCinza, imagemBinaria, limiar, valorMax, tipoThresholdSimples);
                break;
            case LOCAL_MEDIA:
            case LOCAL_GAUSSIANA:

                if (binparams.length < 2) throw new IllegalArgumentException("Binarização ADAPTATIVA requer 2 parâmetros: tamanhoBloco (int) e constanteC (double).");
                int tamanhoBloco = (int) binparams[0];
                double constanteC = (double) binparams[1];
                int metodoAdaptativo = (metodo == LOCAL_MEDIA) ? ADAPTIVE_THRESH_MEAN_C : ADAPTIVE_THRESH_GAUSSIAN_C;

                adaptiveThreshold(imagemCinza, imagemBinaria, valorMax, metodoAdaptativo, THRESH_BINARY, tamanhoBloco, constanteC);
                break;
            default:
                throw new UnsupportedOperationException("Tipo de binarização não implementado: " + metodo);
        }
        return  imagemBinaria;
    }
    public Mat isolarCanal(CanalCor canal, int canalEscolhido) {
        Mat imagemOriginal = this.getImagemPura();
        if (imagemOriginal == null || imagemOriginal.empty()) {
            throw new IllegalStateException("A imagem de origem não pode ser nula ou vazia.");
        }
        if (canal == CanalCor.REVERSE || canal == CanalCor.GRAYSCALE) {
            throw new IllegalArgumentException("REVERSE ou GRAYSCALE não é um espaço de cor válido para isolar canais.");
        }
        if (canalEscolhido < 1 || canalEscolhido > 3) {
            throw new IllegalArgumentException("O canal escolhido deve estar entre 1 e 3 para imagens coloridas.");
        }
        // --- Lógica de Separação de Canais ---
        // 1. Cria um "vetor" (lista) para armazenar os canais separados
        MatVector canais = new MatVector();
        // 2. A função split() faz a mágica: separa a imagem em seus canais individuais
        split(imagemOriginal, canais);
        int indice = canalEscolhido - 1;
        Mat canalIsolado = canais.get(indice);
        return canalIsolado;
    }
}
