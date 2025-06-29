from util import *
from math import ceil, log2
import numpy as np
import random

class AdvancedCryptographicKeyMaterialContainer:
    def __init__(self, ring_dimension_parameter, field_characteristic_modulus, 
                 algebraic_structure_representation, secret_key_vector, 
                 public_key_matrix, binary_decomposition_parameter, 
                 extended_matrix_dimension, gaussian_error_variance):
        self.ring_dimension_parameter = ring_dimension_parameter
        self.field_characteristic_modulus = field_characteristic_modulus
        self.algebraic_structure_representation = algebraic_structure_representation
        self.secret_key_vector = secret_key_vector
        self.public_key_matrix = public_key_matrix
        self.binary_decomposition_parameter = binary_decomposition_parameter
        self.extended_matrix_dimension = extended_matrix_dimension
        self.gaussian_error_variance = gaussian_error_variance
        self.chinese_remainder_theorem_component_count = algebraic_structure_representation.chinese_remainder_theorem_component_count

class ParallelCryptographicComputationOrchestrator:
    @staticmethod
    def execute_parallel_chinese_remainder_theorem_operation(ring_element_collection, computational_operation):
        if not ring_element_collection:
            return []
        
        chinese_remainder_decomposition_registry = []
        for algebraic_element in ring_element_collection:
            if hasattr(algebraic_element, 'decompose_via_chinese_remainder_theorem'):
                chinese_remainder_decomposition_registry.append(
                    algebraic_element.decompose_via_chinese_remainder_theorem()
                )
            else:
                if isinstance(algebraic_element, (list, np.ndarray)):
                    decomposition_matrix = []
                    for matrix_row in algebraic_element:
                        if hasattr(matrix_row, 'decompose_via_chinese_remainder_theorem'):
                            decomposition_matrix.append(
                                matrix_row.decompose_via_chinese_remainder_theorem()
                            )
                        else:
                            decomposition_row = []
                            for matrix_element in matrix_row:
                                if hasattr(matrix_element, 'decompose_via_chinese_remainder_theorem'):
                                    decomposition_row.append(
                                        matrix_element.decompose_via_chinese_remainder_theorem()
                                    )
                                else:
                                    decomposition_row.append(matrix_element)
                            decomposition_matrix.append(decomposition_row)
                    chinese_remainder_decomposition_registry.append(decomposition_matrix)
                else:
                    chinese_remainder_decomposition_registry.append(algebraic_element)
        
        computational_results = []
        component_count = (ring_element_collection[0].ambient_algebraic_structure.chinese_remainder_theorem_component_count 
                          if hasattr(ring_element_collection[0], 'ambient_algebraic_structure') else 1)
        
        for component_index in range(component_count):
            component_arguments = []
            for element_decomposition in chinese_remainder_decomposition_registry:
                if isinstance(element_decomposition, list) and len(element_decomposition) > component_index:
                    component_arguments.append(element_decomposition[component_index])
                else:
                    component_arguments.append(element_decomposition)
            
            component_result = computational_operation(*component_arguments)
            computational_results.append(component_result)
        
        return computational_results

class HierarchicalCryptographicKeyGenerationEngine:
    def __init__(self):
        self.cryptographic_prime_oracle = CryptographicPrimeGenerationOracle()
        self.cyclotomic_ring_constructor = CyclotomicPolynomialRingConstructor()
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
    
    def synthesize_advanced_cryptographic_key_material(self, ring_dimension_specification, 
                                                      modulus_bit_length_parameter=None):
        self._validate_cryptographic_parameters(ring_dimension_specification, modulus_bit_length_parameter)
        
        if modulus_bit_length_parameter is None:
            modulus_bit_length_parameter = ring_dimension_specification + 8
        
        self._validate_modulus_specifications(modulus_bit_length_parameter)
        
        stat("Initiating sophisticated modulus generation protocol")
        field_characteristic_modulus = self.cryptographic_prime_oracle.generate_sophie_germain_prime_configuration(
            modulus_bit_length_parameter
        )
        
        binary_decomposition_parameter = ceil(log2(field_characteristic_modulus))
        extended_matrix_dimension = 2 * binary_decomposition_parameter
        
        self._display_cryptographic_parameters(field_characteristic_modulus, ring_dimension_specification, 
                                              binary_decomposition_parameter)
        
        stat("Constructing advanced algebraic structure representation")
        algebraic_structure = self.cyclotomic_ring_constructor.construct_cyclotomic_ring_structure(
            ring_dimension_specification, field_characteristic_modulus
        )
        
        self._display_algebraic_structure_information(algebraic_structure, ring_dimension_specification, 
                                                     field_characteristic_modulus)
        
        stat("Generating sophisticated secret key material")
        secret_key_vector = self._generate_secret_key_vector(algebraic_structure)
        
        stat("Constructing public key infrastructure")
        public_key_matrix = self._generate_public_key_matrix(algebraic_structure, secret_key_vector)
        
        stat("Executing cryptographic key material verification protocol")
        self._verify_key_material_correctness(secret_key_vector, public_key_matrix, algebraic_structure)
        
        stat("Advanced cryptographic key generation protocol completed successfully")
        
        return AdvancedCryptographicKeyMaterialContainer(
            ring_dimension_specification, field_characteristic_modulus, algebraic_structure,
            secret_key_vector, public_key_matrix, binary_decomposition_parameter,
            extended_matrix_dimension, 1.0
        )
    
    def _validate_cryptographic_parameters(self, ring_dimension, modulus_bit_length):
        if ring_dimension <= 0 or ring_dimension > 32768:
            raise ValueError("Ring dimension parameter must be within interval [1, 32768]")
    
    def _validate_modulus_specifications(self, modulus_bit_length):
        if modulus_bit_length < 8 or modulus_bit_length > 512:
            raise ValueError("Modulus bit length specification must be within interval [8, 512]")
    
    def _display_cryptographic_parameters(self, modulus, ring_dimension, binary_decomposition):
        print(" "*12 + "Field characteristic modulus = %d" % modulus)
        print(" "*12 + "Ring dimension parameter = %d" % ring_dimension)
        print(" "*12 + "Binary decomposition parameter = %d" % binary_decomposition)
    
    def _display_algebraic_structure_information(self, structure, dimension, modulus):
        print(" "*12 + "Algebraic structure: Z_%d[x]/(x^%d + 1)" % (modulus, dimension))
        print(" "*12 + "Chinese remainder theorem components: %d" % structure.chinese_remainder_theorem_component_count)
    
    def _generate_secret_key_vector(self, algebraic_structure):
        gaussian_error_variance = 1.0
        secret_polynomial_element = algebraic_structure.construct_additive_identity_element().generate_discrete_gaussian_error_instance(gaussian_error_variance)
        
        multiplicative_identity = algebraic_structure.construct_multiplicative_identity_element()
        additive_inverse_secret = -secret_polynomial_element
        
        return [multiplicative_identity, additive_inverse_secret]
    
    def _generate_public_key_matrix(self, algebraic_structure, secret_key_vector):
        uniformly_distributed_element = algebraic_structure.generate_uniformly_distributed_element()
        
        gaussian_error_variance = 1.0
        error_polynomial_element = algebraic_structure.construct_additive_identity_element().generate_discrete_gaussian_error_instance(gaussian_error_variance)
        
        public_key_first_component = uniformly_distributed_element * secret_key_vector[1] + error_polynomial_element
        
        return [public_key_first_component, uniformly_distributed_element]
    
    def _verify_key_material_correctness(self, secret_key, public_key, algebraic_structure):
        verification_computation = (secret_key[0] * public_key[0] + 
                                   secret_key[1] * public_key[1])
        expected_error_component = public_key[0] + secret_key[1] * public_key[1]

def keygen(ring_dimension_specification, modulus_bit_length_parameter=None):
    key_generation_engine = HierarchicalCryptographicKeyGenerationEngine()
    return key_generation_engine.synthesize_advanced_cryptographic_key_material(
        ring_dimension_specification, modulus_bit_length_parameter
    )

if __name__ == '__main__':
    advanced_key_material = keygen(16, 24)
    
    print("\nAdvanced cryptographic key structure analysis:")
    print("Secret key vector s = [1, -t]:")
    print("  s[0] =", advanced_key_material.secret_key_vector[0])
    print("  s[1] =", advanced_key_material.secret_key_vector[1])
    
    print("\nPublic key matrix A = [b, a]:")
    print("  A[0] =", advanced_key_material.public_key_matrix[0])
    print("  A[1] =", advanced_key_material.public_key_matrix[1])
    
    print("\nChinese remainder theorem decomposition verification:")
    test_algebraic_element = advanced_key_material.algebraic_structure_representation.generate_uniformly_distributed_element()
    decomposition_components = test_algebraic_element.decompose_via_chinese_remainder_theorem()
    print("  Algebraic element:", test_algebraic_element)
    print("  CRT decomposition:", decomposition_components)
    
    reconstructed_element = AlgebraicElementRepresentation.reconstruct_from_chinese_remainder_decomposition(
        advanced_key_material.algebraic_structure_representation, decomposition_components
    )
    print("  Reconstructed element:", reconstructed_element)
    