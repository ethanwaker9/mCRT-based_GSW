# mCRT-based_GSW

This repository implements the Gentry-Sahai-Waters (GSW) Fully Homomorphic Encryption scheme, utilizing optimizations based on the multi-Chinese Remainder Theorem Ring Learning With Errors (mCRT-RLWE) approach. 

The architecture has been adapted from [Hedglin et al. MIT 2019](https://github.com/hedglinnolan/GSW-Homomorphic-Encryption-Python). This project is a minimal implementation for research purposes. For any real-world deployment or practical development, consider strengthening the information flow in the code and increasing the security parameters.  

-----------------------------

**Core Components**
- util.py: polynomial ring structures with CRT decomposition
- keygen.py: Hierarchical key material synthesis  
- enc.py: homomorphic encryption orchestration
- dec.py: ciphertext decryption with error resilience
- gsw.py: comprehensive verification protocols
- benchmark.py: performance analysis and optimization framework

-----------------------------

**Features**
- Security Level: 128-bit (post-quantum sec)
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

Performance Benchmarking:
```
python benchmark.py
```

Comprehensive Testing:
```
python gsw.py
```

-----------------------------

**Configuration**

Parameter Scaling Options

Lightweight Configuration:
```
keys_light = keygen(2**8, 32)    # Fast, development use
```

Intermediate Configuration:  
```
keys_medium = keygen(2**10, 64)  # Balanced performance/security
```

Production Configuration:
```
keys_heavy = keygen(2**12, 109)  # High security, slower performance
```

Error Distribution Configuration:
- Standard: sigma = 3.2
- Low noise: sigma = 1.0  
- High precision: sigma = 8/sqrt(2*pi)

-----------------------------

**Procedure and Structure**

Algebraic Structure Hierarchy:
- AbstractAlgebraicStructureRepresentation
  - CyclotomicPolynomialRingConstructor
  - AlgebraicElementRepresentation
  - ChineseRemainderTheoremDecomposition
  - BinaryDecompositionMatrixGenerator

Encryption Orchestration:
- AdvancedHomomorphicEncryptionOrchestrator
  - ParallelChineseRemainderTheoremEncryptionEngine
  - AdvancedHomomorphicOperationProcessor
  - BinaryDecompositionCiphertextProcessor
  - AdvancedHomomorphicMultiplicationEngine

Benchmarking Framework:
- MasterBenchmarkingOrchestrationEngine
  - AdvancedParameterSetBenchmarkingOrchestrator
  - ComprehensiveBenchmarkResultsPresentationEngine
  - AdvancedComparativeAnalysisEngine
  - SystemInformationReportingEngine

-----------------------------

**Testing and Validation**

Correctness Verification:
```
python gsw.py
# Executes comprehensive cryptographic verification protocols
```

Performance Analysis:
```
python benchmark.py
# Generates detailed performance metrics and analysis
```

Individual Component Testing:
```
from keygen import keygen
from enc import encrypt
from dec import decrypt
keys = keygen(256, 32)
ciphertext = encrypt(keys, 42)
plaintext = decrypt(keys, ciphertext)
assert plaintext == 42
```

-----------------------------

**Security Considerations**

Parameter Selection Guidelines:
- Development: n=2^8, log q=32 (fast, low security)
- Testing: n=2^10, log q=64 (moderate security)  
- Production: n=2^12, log q=109+ (high security)

-----------------------------

**References and Academic Foundations**
This implementation is based on research in:
- GSW Cryptosystem: [Gentry, Sahai, Waters (2013)](https://link.springer.com/chapter/10.1007/978-3-642-40041-4_5)
- Ring-LWE: [Lyubashevsky, Peikert, Regev (2010)](https://link.springer.com/chapter/10.1007/978-3-642-13190-5_1)
- Homomorphic Encryption: [Gentry's breakthrough work (2009)](https://link.springer.com/chapter/10.1007/978-3-642-13190-5_2)
