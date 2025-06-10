import hashlib
import os
from cryptography.fernet import Fernet
import torch


def hash_model_parameters(model):
    """Compute SHA-256 hash of all model parameters."""
    state_bytes = b''.join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
    return hashlib.sha256(state_bytes).hexdigest()


def generate_secret_key(path="secret.key"):
    """Generate and save a Fernet secret key (only once)."""
    if not os.path.exists(path):
        key = Fernet.generate_key()
        with open(path, 'wb') as f:
            f.write(key)


def load_secret_key(path="secret.key"):
    """Load the saved Fernet key."""
    with open(path, 'rb') as f:
        return f.read()


def encrypt_hash(hash_str, secret_key):
    """Encrypt a model hash string with a secret key."""
    fernet = Fernet(secret_key)
    return fernet.encrypt(hash_str.encode())


def decrypt_hash(encrypted_hash, secret_key):
    """Decrypt a hash string."""
    fernet = Fernet(secret_key)
    return fernet.decrypt(encrypted_hash).decode()


def save_model_with_encrypted_hash(model, path="lyapunov_model.pt", encrypted_hash_path="model_hash.enc"):
    torch.save(model.state_dict(), path)
    hash_val = hash_model_parameters(model)

    generate_secret_key()
    secret_key = load_secret_key()
    encrypted_hash = encrypt_hash(hash_val, secret_key)

    with open(encrypted_hash_path, 'wb') as f:
        f.write(encrypted_hash)


def verify_model_integrity_encrypted(model, encrypted_hash_path="model_hash.enc"):
    current_hash = hash_model_parameters(model)
    secret_key = load_secret_key()

    if not os.path.exists(encrypted_hash_path):
        print("Warning: No encrypted hash found.")
        return False

    with open(encrypted_hash_path, 'rb') as f:
        encrypted_hash = f.read()

    decrypted_hash = decrypt_hash(encrypted_hash, secret_key)
    return decrypted_hash == current_hash

def save_obfuscated_model(model, path="encrypted_model.pt"):
    scripted = torch.jit.script(model)
    scripted.save(path)