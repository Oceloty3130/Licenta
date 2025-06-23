import hashlib
import os
import pathlib

from cryptography.fernet import Fernet
import torch

def generate_secret_key(folder_path):
    path = os.path.join(folder_path, "secret.key")
    if not os.path.exists(path):
        key = Fernet.generate_key()
        with open(path, 'wb') as f:
            f.write(key)


def load_secret_key(folder_path):
    path = os.path.join(folder_path, "secret.key")
    with open(path, 'rb') as f:
        return f.read()

def hash_model_parameters(model):
    state_bytes = b''.join([p.detach().cpu().numpy().tobytes() for p in model.parameters()])
    return hashlib.sha256(state_bytes).hexdigest()

def encrypt_hash(hash_str, secret_key):
    fernet = Fernet(secret_key)
    return fernet.encrypt(hash_str.encode())


def decrypt_hash(encrypted_hash, secret_key):
    fernet = Fernet(secret_key)
    return fernet.decrypt(encrypted_hash).decode()


def save_model_with_encrypted_hash(model, path):
    folder_path = os.path.join(path, "Model_Parameters")
    os.makedirs(folder_path, exist_ok=True)

    parameters_path = os.path.join(folder_path, "lyapunov_model.pt")
    torch.save(model.state_dict(), parameters_path)

    hash_val = hash_model_parameters(model)
    generate_secret_key(folder_path)
    secret_key = load_secret_key(folder_path)
    encrypted_hash = encrypt_hash(hash_val, secret_key)
    with open(os.path.join(folder_path, "model_hash.enc"), 'wb') as f:
        f.write(encrypted_hash)

    save_obfuscated_model(model, path)


def verify_model_integrity_encrypted(model, folder_path):
    encrypted_hash_path = os.path.join(folder_path, "model_hash.enc")

    if os.path.exists(encrypted_hash_path):
        if not os.path.exists(os.path.join(folder_path, "secret.key")):
            return False

        current_hash = hash_model_parameters(model)
        secret_key = load_secret_key(folder_path)

        with open(encrypted_hash_path, 'rb') as f:
            encrypted_hash = f.read()

        decrypted_hash = decrypt_hash(encrypted_hash, secret_key)
        return decrypted_hash == current_hash
    else:
        print("Warning: No encrypted hash found.")
        return False

def save_obfuscated_model(model, path):
    folder_path = os.path.join(path, "Model_Parameters")
    encrypt_path = os.path.join(folder_path, "encrypted_model.pt")
    scripted = torch.jit.script(model)
    scripted.save(encrypt_path)

def load_model(path, model, device, input_size, hidden_layer_size, alpha):
    folder_path = os.path.join(path, "Model_Parameters")
    model_path = os.path.join(folder_path, "lyapunov_model.pt")

    if os.path.exists(model_path):
        try:
            loaded_state_dict = torch.load(model_path, map_location=device)

            model_state_dict = model.state_dict()
            compatible_state_dict = {
                k: v for k, v in loaded_state_dict.items()
                if k in model_state_dict and v.size() == model_state_dict[k].size()
            }

            if len(compatible_state_dict) != len(model_state_dict):
                print("Model incompatibil (dimensiuni diferite). Antrenează unul nou.")
                return model.__class__(input_size, hidden_layer_size, alpha).to(device)

            model.load_state_dict(compatible_state_dict)
            model.eval()
            print("Model loaded.")
        except Exception as e:
            print(f"Nu s-a putut încărca modelul: {e}")
            print("Se antrenează un model nou.")
            return model.__class__(input_size, hidden_layer_size, alpha).to(device)
    else:
        print("Fișier model inexistent. Se antrenează unul nou.")
        return model.__class__(input_size, hidden_layer_size, alpha).to(device)

    if os.path.exists(folder_path):
        if verify_model_integrity_encrypted(model, folder_path):
            print("Model integrity OK.")
        else:
            print("WARNING: Integritatea modelului compromisă.")
            print("Training a new model.")
            return model.__class__(input_size, hidden_layer_size, alpha).to(device)

    return model.to(device)

