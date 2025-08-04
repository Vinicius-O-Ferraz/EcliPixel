import enums.Thresh;
import fi.iki.elonen.NanoHTTPD;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imencode;

/**
 * A Camada de API. Expõe a funcionalidade de processamento de imagem
 * via requisições HTTP.
 */
public class Servidor extends NanoHTTPD {

    // O servidor "possui" o mestre para delegar tarefas em lote.
    private final PixelMestre mestre = new PixelMestre();

    public Servidor() throws IOException {
        super(8080);
        start(NanoHTTPD.SOCKET_READ_TIMEOUT, false);
        System.out.println("\n>>> Servidor rodando! Acesse http://localhost:8080/\n");
    }

    @Override
    public Response serve(IHTTPSession session) {
        String uri = session.getUri();
        Method method = session.getMethod();

        try {
            if (Method.POST.equals(method) && "/processar-agora".equals(uri)) {
                return processarImagemUnicaSincrono(session);
            }
            if (Method.POST.equals(method) && "/iniciar-lote".equals(uri)) {
                return iniciarProcessamentoEmLoteAssincrono(session);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return newFixedLengthResponse(Response.Status.INTERNAL_ERROR, "text/plain", "Erro interno no servidor: " + e.getMessage());
        }
        return newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Endpoint não encontrado.");
    }

    /**
     * Lida com requisições rápidas e síncronas. O cliente envia uma imagem
     * e recebe a imagem processada de volta imediatamente.
     */
    private Response processarImagemUnicaSincrono(IHTTPSession session) throws IOException, ResponseException {
        Map<String, String> files = new HashMap<>();
        // O NanoHTTPD salva o corpo do POST em um arquivo temporário
        session.parseBody(files);
        String tempFilePathStr = files.get("imagem");

        if (tempFilePathStr == null) {
            return newFixedLengthResponse(Response.Status.BAD_REQUEST, "text/plain", "Campo 'imagem' não encontrado no formulário de upload.");
        }

        Path tempPath = Paths.get(tempFilePathStr);
        try {
            // Usa o Correio para ler a imagem
            Mat imagemOriginal = PixelCorreio.lerImagem(tempPath.toString());

            // Usa o Serviço diretamente para o processamento
            Mat imagemProcessada = EcliPixel.binarizar(imagemOriginal, Thresh.OTSU);

            // Codifica o resultado para PNG em memória
            byte[] buffer = new byte[(int) (imagemProcessada.total() * imagemProcessada.channels() * 2)];
            imencode(".png", imagemProcessada, buffer);

            // Retorna a imagem processada na resposta
            return newChunkedResponse(Response.Status.OK, "image/png", new ByteArrayInputStream(buffer));

        } finally {
            // É uma boa prática limpar o arquivo temporário criado pelo NanoHTTPD
            Files.deleteIfExists(tempPath);
        }
    }

    /**
     * Lida com requisições longas e assíncronas. Dispara um trabalho em lote
     * e retorna uma resposta imediata para o cliente.
     */
    private Response iniciarProcessamentoEmLoteAssincrono(IHTTPSession session) {
        Map<String, String> params = session.getParms();
        String pastaEntrada = params.get("entrada");
        String pastaSaida = params.get("saida");

        if (pastaEntrada == null || pastaSaida == null) {
            return newFixedLengthResponse(Response.Status.BAD_REQUEST, "text/plain", "Parâmetros de URL 'entrada' e 'saida' são obrigatórios.");
        }

        // DELEGA o trabalho pesado para o Mestre em uma nova thread,
        // para que a requisição HTTP possa retornar imediatamente.
        new Thread(() -> {
            mestre.executarBinarizacaoOtimizadaEmLote(pastaEntrada, pastaSaida);
        }).start();

        // Retorna a resposta 202 Accepted, significando "Seu pedido foi aceito e está sendo processado".
        String mensagem = "Processamento em lote iniciado. Entrada: " + pastaEntrada + ", Saída: " + pastaSaida;
        return newFixedLengthResponse(Response.Status.ACCEPTED, "text/plain", mensagem);
    }

    public static void main(String[] args) {
        try {
            new Servidor();
        } catch (IOException ioe) {
            System.err.println("Não foi possível iniciar o servidor:\n" + ioe);
        }
    }
}
