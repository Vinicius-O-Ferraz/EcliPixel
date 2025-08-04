package exceptions;

public class FalhaAoCarregarImagem extends RuntimeException {
    public FalhaAoCarregarImagem(String message) {
        super(message);
    }
}
