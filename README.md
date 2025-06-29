# mCRT-based_GSW

This repository implements the Gentry-Sahai-Waters (GSW) Fully Homomorphic Encryption scheme, utilizing optimizations based on the multi-Chinese Remainder Theorem Ring Learning With Errors (mCRT-RLWE) approach. 

The architecture has been adapted from [Hedglin et al. MIT 2019](https://github.com/hedglinnolan/GSW-Homomorphic-Encryption-Python). This project is a minimal implementation for research purposes. For any real-world deployment or practical development, consider strengthening the information flow in the code and increasing the security parameters.  

-----------------------------

**Core Components**
- util.py: Advanced polynomial ring structures with CRT decomposition
- keygen.py: Hierarchical cryptographic key material synthesis  
- enc.py: Sophisticated homomorphic encryption orchestration
- dec.py: Advanced ciphertext decryption with error resilience
- gsw.py: Comprehensive cryptographic verification protocols
- benchmark.py: Performance analysis and optimization framework

-----------------------------

**Features**
- Security Level: 128-bit post-quantum security
- Parallel Processing: Chinese Remainder Theorem optimization
- Homomorphic Operations: Addition, scalar multiplication, and composition
- Scalable Parameters: Configurable ring dimensions and modulus sizes
- Error Management: Discrete Gaussian error distribution with configurable variance

-----------------------------

**Usage**

Key Generation:
```
from keygen import keygen
keys = keygen(ring_dimension=256, modulus_bits=32)
```

Encryption:
```
from enc import encrypt
ciphertext_a = encrypt(keys, 15)
ciphertext_b = encrypt(keys, 25)
```

Homomorphic Operations:
```
from enc import homomorphic_add, homomorphic_mult_by_scalar
homomorphic_sum = homomorphic_add(keys, ciphertext_a, ciphertext_b)
scaled_result = homomorphic_mult_by_scalar(keys, homomorphic_sum, 3)
```

Decryption:
```
from dec import decrypt
result = decrypt(keys, scaled_result)
```

-----------------------------

**Execution Instruction**

Hi
