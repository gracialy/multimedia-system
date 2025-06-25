from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from collections import Counter
import heapq
import os 

QUANTIZATION_TABLE: np.ndarray = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def get_8x8_block(image_path: str, x_start: int, y_start: int) -> np.ndarray:
    """
    Loads an image, converts it to YCbCr, and extracts a single 8x8 luminance (Y) block.
    """
    img: Image.Image = Image.open(image_path).convert('YCbCr')
    img_array: np.ndarray = np.array(img)
    y_block: np.ndarray = img_array[y_start : y_start + 8, x_start : x_start + 8, 0]
    return y_block

def apply_dct_2d(block: np.ndarray) -> np.ndarray:
    """
    Applies the 2D Discrete Cosine Transform to an 8x8 block.
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct_2d(dct_block: np.ndarray) -> np.ndarray:
    """
    Applies the 2D Inverse Discrete Cosine Transform to a block of DCT coefficients.
    """
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

def quantize_block(dct_coeffs: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """
    Quantizes the DCT coefficients.
    """
    return np.round(dct_coeffs / q_table).astype(np.int8)

def dequantize_block(quantized_block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """
    Dequantizes the coefficients.
    """
    return quantized_block * q_table

def zig_zag_scan(block: np.ndarray) -> np.ndarray:
    """
    Performs a zig-zag scan on an 8x8 block to convert it to a 1D array.
    """
    rows, cols = block.shape
    output = np.zeros(rows * cols, dtype=block.dtype)
    
    i, j = 0, 0
    up = True 
    
    for k in range(rows * cols):
        output[k] = block[i, j]
        
        if up:
            if j == cols - 1: 
                i += 1
                up = False
            elif i == 0: 
                j += 1
                up = False
            else: 
                i -= 1
                j += 1
        else:
            if i == rows - 1: 
                j += 1
                up = True
            elif j == 0: 
                i += 1
                up = True
            else: 
                i += 1
                j -= 1
                
    return output

def anti_zig_zag_scan(arr: np.ndarray, rows: int = 8, cols: int = 8) -> np.ndarray:
    """
    Performs an inverse zig-zag scan to convert a 1D array back to an 8x8 block.
    """
    block = np.zeros((rows, cols), dtype=arr.dtype)
    
    i, j = 0, 0
    up = True
    
    for k in range(rows * cols):
        block[i, j] = arr[k]
        
        if up:
            if j == cols - 1:
                i += 1
                up = False
            elif i == 0:
                j += 1
                up = False
            else:
                i -= 1
                j += 1
        else:
            if i == rows - 1:
                j += 1
                up = True
            elif j == 0:
                i += 1
                up = True
            else:
                i += 1
                j -= 1
                
    return block

class Node:
    def __init__(self, char: Optional[int], freq: int, left: 'Optional[Node]' = None, right: 'Optional[Node]' = None):
        self.char = char 
        self.freq = freq 
        self.left = left 
        self.right = right 

    def __lt__(self, other: 'Node') -> bool:
        return self.freq < other.freq

def build_huffman_tree_and_codes(data: np.ndarray) -> Tuple[Dict[int, str], Node]:
    """
    Builds a Huffman tree and generates Huffman codes for the given 1D data array.
    """
    frequency: Dict[int, int] = Counter(data.flatten())
    
    pq: List[Node] = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(pq) 

    while len(pq) > 1:
        left = heapq.heappop(pq)
        right = heapq.heappop(pq)
        merged = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(pq, merged)
    
    root: Node = pq[0]

    huffman_codes: Dict[int, str] = {}

    def _generate_codes(node: Node, current_code: str):
        if node.char is not None:
            huffman_codes[node.char] = current_code
            return
        if node.left:
            _generate_codes(node.left, current_code + '0')
        if node.right:
            _generate_codes(node.right, current_code + '1')
    
    _generate_codes(root, "")
    
    return huffman_codes, root

def huffman_encode(data: np.ndarray, huffman_codes: Dict[int, str]) -> str:
    """
    Encodes the 1D data array using the generated Huffman codes. Returns a string of concatenated binary codes.
    """
    encoded_bits: str = ""
    for value in data:
        encoded_bits += huffman_codes[value]
    return encoded_bits

def huffman_decode(encoded_bits: str, huffman_tree_root: Node) -> np.ndarray:
    """
    Decodes the binary string using the Huffman tree to reconstruct the original data.
    """
    decoded_values: List[int] = []
    current_node: Node = huffman_tree_root
    
    for bit in encoded_bits:
        if bit == '0':
            current_node = current_node.left 
        else: 
            current_node = current_node.right 
        
        if current_node.char is not None:
            decoded_values.append(current_node.char)
            current_node = huffman_tree_root
            
    return np.array(decoded_values, dtype=np.int8)

def save_block_as_image(block: np.ndarray, filename: str, title: str, cmap: str = 'gray', vmin: int = 0, vmax: int = 255):
    """
    Saves a single 8x8 block as a grayscale image.
    Automatically creates 'output_images' directory if it doesn't exist.
    """
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(2, 2)) 
    plt.imshow(block, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.1)
    plt.close() 

def normalize_and_save_coefficients(coeffs: np.ndarray, filename: str, title: str):
    """
    Normalizes DCT/Quantized coefficients to 0-255 range and saves them as a grayscale image.
    """
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    normalized_coeffs = coeffs.copy().astype(np.float32) 
    
    if np.all(normalized_coeffs == 0):
        normalized_coeffs = np.zeros_like(normalized_coeffs)
    else:
        normalized_coeffs = (normalized_coeffs - normalized_coeffs.min()) / (normalized_coeffs.max() - normalized_coeffs.min())
        normalized_coeffs = (normalized_coeffs * 255).astype(np.uint8)

    plt.figure(figsize=(2, 2))
    plt.imshow(normalized_coeffs, cmap='gray', vmin=0, vmax=255)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.1)
    plt.close()

def compare_blocks_and_highlight_diff(original: np.ndarray, reconstructed: np.ndarray, mse: float, psnr: float | str, filename: str):
    """
    Creates a single image comparing the original, reconstructed, and their absolute difference. Highlights the differences visually.
    """
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    
    if diff.max() > 0:
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    else:
        diff_normalized = np.zeros_like(diff, dtype=np.uint8) 

    plt.figure(figsize=(10, 3)) 

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Block", fontsize=12)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Reconstructed Block\n(MSE: {mse:.2f}, PSNR: {psnr:.2f} dB)", fontsize=12)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(diff_normalized, cmap='hot', vmin=0, vmax=255) 
    plt.title("Absolute Difference (Higher = More Diff)", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    image_path: str = 'selfie.jpg' 
    x_coord: int = 0  
    y_coord: int = 0  
    
    print("--- JPEG Encoding Simulation (for one 8x8 Luminance Block) ---")
    
    # 1. Konversi Ruang Warna (RGB ke YCbCr) & Pembagian Blok
    original_luminance_block: np.ndarray = get_8x8_block(image_path, x_coord, y_coord)
    print(f"\n[Step 1 & 3: Original 8x8 Luminance Block extracted.]")
    save_block_as_image(original_luminance_block, "1_original_block.png", "Original Block")

    # 4. Pergeseran Tingkat (Level Shifting)
    shifted_block_for_dct: np.ndarray = original_luminance_block.astype(np.float32) - 128
    print(f"[Step 4: Level Shifted Block completed.]")

    # 5. Transformasi Kosinus Diskrit (DCT)
    dct_coefficients: np.ndarray = apply_dct_2d(shifted_block_for_dct)
    print(f"[Step 5: DCT Coefficients calculated.]")
    normalize_and_save_coefficients(dct_coefficients, "2_dct_coefficients.png", "DCT Coefficients")

    # 6. Kuantisasi
    quantized_coefficients: np.ndarray = quantize_block(dct_coefficients, QUANTIZATION_TABLE)
    print(f"[Step 6: Quantized Coefficients calculated.]")
    normalize_and_save_coefficients(quantized_coefficients, "3_quantized_coefficients.png", "Quantized Coefficients")
    
    # 7. Pengurutan Zig-Zag
    zig_zag_array: np.ndarray = zig_zag_scan(quantized_coefficients)
    print(f"[Step 7: Zig-Zag Ordered Array generated.]")
    
    # 8. Pengkodean Entropy (Huffman)
    huffman_codes, huffman_tree_root = build_huffman_tree_and_codes(zig_zag_array)
    encoded_bitstream: str = huffman_encode(zig_zag_array, huffman_codes)
    print(f"[Step 8: Huffman Encoding completed. Encoded bitstream length: {len(encoded_bitstream)} bits]")
    
    # --- JPEG Decoding Simulation ---
    print("\n--- JPEG Decoding Simulation ---")

    # 1. Dekode Entropy (Huffman)
    decoded_zig_zag_array: np.ndarray = huffman_decode(encoded_bitstream, huffman_tree_root)
    print(f"[Step 1: Huffman Decoding completed.]")
    
    # 2. Anti-Pengurutan Zig-Zag
    de_zig_zag_block: np.ndarray = anti_zig_zag_scan(decoded_zig_zag_array)
    print(f"[Step 2: De-Zig-Zagged Block generated.]")

    # 3. Dequantisasi
    dequantized_coefficients: np.ndarray = dequantize_block(de_zig_zag_block, QUANTIZATION_TABLE)
    print(f"[Step 3: Dequantized Coefficients calculated.]")

    # 4. Inversi DCT (IDCT)
    reconstructed_shifted_block: np.ndarray = apply_idct_2d(dequantized_coefficients)
    print(f"[Step 4: Reconstructed Shifted Block calculated.]")

    # 5. Inversi Pergeseran Tingkat
    reconstructed_luminance_block: np.ndarray = reconstructed_shifted_block + 128
    reconstructed_luminance_block = np.clip(reconstructed_luminance_block, 0, 255).astype(np.uint8)
    print(f"[Step 5: Final Reconstructed Luminance Block calculated.]")
    save_block_as_image(reconstructed_luminance_block, "4_reconstructed_block.png", "Reconstructed Block")
    
    # --- Analysis and Final Display ---
    print("\n--- Analysis ---")
    
    mse: float = np.mean((original_luminance_block - reconstructed_luminance_block) ** 2)
    print(f"\nMean Squared Error (MSE): {mse:.2f}")

    psnr: float | str
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel_value: float = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")

    compare_blocks_and_highlight_diff(original_luminance_block, reconstructed_luminance_block, mse, psnr, "5_comparison_and_difference.png")
    print("\n[Comparison image saved to 'output_images/5_comparison_and_difference.png']")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Block")
    plt.imshow(original_luminance_block, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Reconstructed Block\nMSE: {mse:.2f}, PSNR: {psnr:.2f} dB")
    plt.imshow(reconstructed_luminance_block, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("\nProcess completed successfully. Check the 'output_images' folder for all saved visuals.")