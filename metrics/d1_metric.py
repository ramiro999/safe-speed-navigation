import cv2
import numpy as np

def disp_read(filename):
    """Load disparity map from PNG file."""
    I = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if len(I.shape) == 3 and I.shape[-1] > 1:
        I = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY).astype(np.float32)
    D = I.astype(np.float32) / 256.0
    D[I == 0] = -1  # Set invalid pixels
    return D

def disp_write(D, filename):
    """Save disparity map to PNG file."""
    D = np.clip(D * 256, 0, 65535).astype(np.uint16)
    cv2.imwrite(filename, D)

def disp_error(D_gt, D_est, tau):
    """Calculate disparity error."""
    E = np.abs(D_gt - D_est)
    valid_pixels = (D_gt > 0)
    error_pixels = (E > tau[0]) & (E / np.abs(D_gt) > tau[1])
    d_err = np.sum(error_pixels & valid_pixels) / np.sum(valid_pixels)
    return d_err

def disp_error_map(D_gt, D_est):
    """Compute disparity error map."""
    valid_pixels = (D_gt >= 0)
    E = np.abs(D_gt - D_est)
    E[~valid_pixels] = 0  # Set invalid pixels to zero
    return E, valid_pixels

def error_colormap():
    """Define error colormap (RGB values normalized to 0-1 range)."""
    return np.array([
        [0/3.0, 0.1875/3.0, 49/255, 54/255, 149/255],
        [0.1875/3.0, 0.375/3.0, 69/255, 117/255, 180/255],
        [0.375/3.0, 0.75/3.0, 116/255, 173/255, 209/255],
        # ... (continúa con el resto de los colores)
    ])

# Ejemplo de uso
if __name__ == "__main__":
    D_est = disp_read('../disp_est.png')
    D_gt = disp_read('../disp_gt.png')
    D_gt[D_gt < 0] = 0  # Reemplazar valores negativos para evitar errores en cálculos
    #D_est = (D_est / np.max(D_est)) * np.max(D_gt[D_gt > 0]) # Normalizar D_est para que tenga el mismo rango que D_gt (0 a max(D_gt)) 
    D_est[D_est < 0] = -1.0 # Set invalid pixels to -1

    print(f"Shape of D_est: {D_est.shape}")
    print(f"Shape of D_gt: {D_gt.shape}")

    # Error calculation
    tau = [3, 0.05]
    d_err = disp_error(D_gt, D_est, tau)
    print(f"Disparity Error: {d_err :.5f}")
    print("D_gt unique values:", np.unique(D_gt))
    print("D_est unique values:", np.unique(D_est))

    # Convertir a 8 bits para visualización
    # D_est_8bit = cv2.normalize(D_est, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # D_gt_8bit = cv2.normalize(D_gt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert D_est and D_gt to 8-bit grayscale images if they are not already
    D_est_8bit = (D_est / np.max(D_est) * 255).astype(np.uint8)
    D_gt_8bit = (D_gt / np.max(D_gt) * 255).astype(np.uint8)

    # Mostrar las imágenes usando OpenCV
    cv2.imshow('D_est', D_est_8bit)
    cv2.imshow('D_gt', D_gt_8bit)

    cv2.waitKey(0)
    cv2.destroyAllWindows()