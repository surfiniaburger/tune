import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64

# In a real mobile app, this key would be securely managed by
# the Android Keystore or iOS Keychain. For this skeleton, we'll
# use an environment variable for demonstration.
SECRET_KEY = os.environ.get("AURA_MIND_SECRET_KEY", "a_default_secret_key_32_bytes_!!").encode()

if len(SECRET_KEY) != 32:
    raise ValueError("SECRET_KEY must be 32 bytes long for AES-256.")

def encrypt_data(data: bytes) -> bytes:
    """Encrypts data using AES-CBC."""
    iv = os.urandom(16)
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    
    cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return iv + encrypted_data

def decrypt_data(encrypted_data_with_iv: bytes) -> bytes:
    """Decrypts data using AES-CBC."""
    iv = encrypted_data_with_iv[:16]
    encrypted_data = encrypted_data_with_iv[16:]
    
    cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data