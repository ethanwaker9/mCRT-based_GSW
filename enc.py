from util import *
import numpy as np

class AdvancedHomomorphicEncryptionOrchestrator:
    def __init__(self):
        self.binary_decomposition_generator = BinaryDecompositionMatrixGenerator()
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
    
    def execute_sophisticated_encryption_protocol(self, cryptographic_key_material, plaintext_message):
        stat("Initiating sophisticated homomorphic encryption protocol")
        
        normalized_plaintext_element = self._normalize_plaintext_representation(
            cryptographic_key_material, plaintext_message
        )
        
        binary_decomposition_identity_matrix = self.binary_decomposition_generator.construct_binary_decomposition_identity_matrix(
            cryptographic_key_material.extended_matrix_dimension
        )
        
        randomness_vector = self._generate_cryptographic_randomness_vector(cryptographic_key_material)
        error_matrix = self._generate_discrete_gaussian_error_matrix(cryptographic_key_material)
        
        ciphertext_matrix = self._compute_ciphertext_matrix(
            cryptographic_key_material, normalized_plaintext_element,
            binary_decomposition_identity_matrix, randomness_vector, error_matrix
        )
        
        stat("Sophisticated encryption protocol execution completed")
        return ciphertext_matrix
    
    def _normalize_plaintext_representation(self, key_material, plaintext_message):
        if isinstance(plaintext_message, (int, np.integer)):
            coefficient_vector = [plaintext_message] + [0] * (key_material.algebraic_structure_representation.algebraic_degree - 1)
            return key_material.algebraic_structure_representation.instantiate_algebraic_element(coefficient_vector)
        elif hasattr(plaintext_message, 'ambient_algebraic_structure'):
            return plaintext_message
        else:
            return key_material.algebraic_structure_representation.instantiate_algebraic_element(plaintext_message)
    
    def _generate_cryptographic_randomness_vector(self, key_material):
        randomness_vector = []
        for vector_index in range(key_material.extended_matrix_dimension):
            binary_coefficient_vector = np.random.randint(
                0, 2, size=key_material.algebraic_structure_representation.algebraic_degree, dtype=np.int64
            )
            randomness_element = key_material.algebraic_structure_representation.instantiate_algebraic_element(binary_coefficient_vector)
            randomness_vector.append(randomness_element)
        return randomness_vector
    
    def _generate_discrete_gaussian_error_matrix(self, key_material):
        error_matrix = []
        for row_index in range(key_material.extended_matrix_dimension):
            error_row = []
            for column_index in range(2):
                error_element = key_material.algebraic_structure_representation.construct_additive_identity_element().generate_discrete_gaussian_error_instance(
                    key_material.gaussian_error_variance
                )
                error_row.append(error_element)
            error_matrix.append(error_row)
        return error_matrix
    
    def _compute_ciphertext_matrix(self, key_material, plaintext_element, 
                                 binary_decomposition_matrix, randomness_vector, error_matrix):
        ciphertext_matrix = []
        
        for matrix_row_index in range(key_material.extended_matrix_dimension):
            ciphertext_row = []
            for matrix_column_index in range(2):
                ciphertext_component = error_matrix[matrix_row_index][matrix_column_index]
                
                randomness_contribution = (randomness_vector[matrix_row_index] * 
                                         key_material.public_key_matrix[matrix_column_index])
                ciphertext_component = ciphertext_component + randomness_contribution
                
                if binary_decomposition_matrix[matrix_row_index][matrix_column_index] != 0:
                    plaintext_contribution = (plaintext_element * 
                                            binary_decomposition_matrix[matrix_row_index][matrix_column_index])
                    ciphertext_component = ciphertext_component + plaintext_contribution
                
                ciphertext_row.append(ciphertext_component)
            ciphertext_matrix.append(ciphertext_row)
        
        return ciphertext_matrix

class ParallelChineseRemainderTheoremEncryptionEngine:
    def __init__(self):
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
    
    def execute_parallel_chinese_remainder_encryption(self, cryptographic_key_material, plaintext_message):
        stat("Initiating parallel Chinese remainder theorem encryption protocol")
        
        normalized_plaintext_element = self._normalize_plaintext_for_crt_decomposition(
            cryptographic_key_material, plaintext_message
        )
        
        plaintext_crt_decomposition = normalized_plaintext_element.decompose_via_chinese_remainder_theorem()
        
        public_key_crt_decomposition = []
        for public_key_component in cryptographic_key_material.public_key_matrix:
            public_key_crt_decomposition.append(public_key_component.decompose_via_chinese_remainder_theorem())
        
        parallel_ciphertext_components = []
        
        for component_index in range(cryptographic_key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count):
            stat(f"Processing CRT component {component_index + 1}/{cryptographic_key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count}")
            
            component_plaintext = plaintext_crt_decomposition[component_index]
            component_public_key = [public_key_crt_decomposition[0][component_index], 
                                   public_key_crt_decomposition[1][component_index]]
            
            component_binary_decomposition = BinaryDecompositionMatrixGenerator.construct_binary_decomposition_identity_matrix(
                cryptographic_key_material.extended_matrix_dimension
            )
            
            component_randomness = self._generate_component_randomness(
                cryptographic_key_material, component_index
            )
            
            component_error_matrix = self._generate_component_error_matrix(
                cryptographic_key_material, component_index
            )
            
            component_ciphertext = self._compute_component_ciphertext(
                cryptographic_key_material, component_plaintext, component_public_key,
                component_binary_decomposition, component_randomness, component_error_matrix,
                component_index
            )
            
            parallel_ciphertext_components.append(component_ciphertext)
        
        reconstructed_ciphertext = self._reconstruct_ciphertext_from_components(
            cryptographic_key_material, parallel_ciphertext_components
        )
        
        stat("Parallel Chinese remainder theorem encryption completed")
        return reconstructed_ciphertext
    
    def _normalize_plaintext_for_crt_decomposition(self, key_material, plaintext_message):
        if isinstance(plaintext_message, (int, np.integer)):
            coefficient_vector = [plaintext_message] + [0] * (key_material.algebraic_structure_representation.algebraic_degree - 1)
            return key_material.algebraic_structure_representation.instantiate_algebraic_element(coefficient_vector)
        else:
            return plaintext_message
    
    def _generate_component_randomness(self, key_material, component_index):
        component_randomness = []
        for randomness_index in range(key_material.extended_matrix_dimension):
            if (key_material.algebraic_structure_representation.irreducible_factorization_components[component_index] and 
                len(key_material.algebraic_structure_representation.irreducible_factorization_components[component_index]) == 2):
                component_randomness.append(np.random.randint(0, 2))
            else:
                binary_coefficients = np.random.randint(
                    0, 2, size=key_material.algebraic_structure_representation.algebraic_degree, dtype=np.int64
                )
                component_randomness.append(binary_coefficients)
        return component_randomness
    
    def _generate_component_error_matrix(self, key_material, component_index):
        component_error_matrix = []
        for row_index in range(key_material.extended_matrix_dimension):
            error_row = []
            for column_index in range(2):
                if (key_material.algebraic_structure_representation.irreducible_factorization_components[component_index] and 
                    len(key_material.algebraic_structure_representation.irreducible_factorization_components[component_index]) == 2):
                    scalar_error = int(np.rint(np.random.normal(0, key_material.gaussian_error_variance)))
                    error_row.append(scalar_error)
                else:
                    polynomial_error_coefficients = np.rint(
                        np.random.normal(0, key_material.gaussian_error_variance, 
                                       key_material.algebraic_structure_representation.algebraic_degree)
                    ).astype(np.int64)
                    error_row.append(polynomial_error_coefficients)
            component_error_matrix.append(error_row)
        return component_error_matrix
    
    def _compute_component_ciphertext(self, key_material, component_plaintext, component_public_key,
                                    binary_decomposition_matrix, component_randomness, 
                                    component_error_matrix, component_index):
        component_ciphertext = []
        for row_index in range(key_material.extended_matrix_dimension):
            ciphertext_row = []
            for column_index in range(2):
                if (key_material.algebraic_structure_representation.irreducible_factorization_components[component_index] and 
                    len(key_material.algebraic_structure_representation.irreducible_factorization_components[component_index]) == 2):
                    ciphertext_element = component_error_matrix[row_index][column_index]
                    ciphertext_element = (ciphertext_element + 
                                        component_randomness[row_index] * component_public_key[column_index]) % key_material.field_characteristic_modulus
                    if binary_decomposition_matrix[row_index][column_index] != 0:
                        ciphertext_element = (ciphertext_element + 
                                            component_plaintext * binary_decomposition_matrix[row_index][column_index]) % key_material.field_characteristic_modulus
                    ciphertext_row.append(ciphertext_element)
                else:
                    ciphertext_element = component_error_matrix[row_index][column_index].copy()
                    ciphertext_row.append(ciphertext_element)
            component_ciphertext.append(ciphertext_row)
        return component_ciphertext
    
    def _reconstruct_ciphertext_from_components(self, key_material, ciphertext_components):
        reconstructed_ciphertext = []
        for row_index in range(key_material.extended_matrix_dimension):
            ciphertext_row = []
            for column_index in range(2):
                component_collection = []
                for component_index in range(key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count):
                    component_collection.append(ciphertext_components[component_index][row_index][column_index])
                
                reconstructed_element = AlgebraicElementRepresentation.reconstruct_from_chinese_remainder_decomposition(
                    key_material.algebraic_structure_representation, component_collection
                )
                ciphertext_row.append(reconstructed_element)
            reconstructed_ciphertext.append(ciphertext_row)
        return reconstructed_ciphertext

class AdvancedHomomorphicOperationProcessor:
    def __init__(self):
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
    
    def execute_homomorphic_addition_protocol(self, cryptographic_key_material, 
                                            first_ciphertext_matrix, second_ciphertext_matrix):
        stat("Executing sophisticated homomorphic addition protocol")
        
        addition_result_matrix = []
        for row_index in range(len(first_ciphertext_matrix)):
            result_row = []
            for column_index in range(len(first_ciphertext_matrix[row_index])):
                additive_combination = (first_ciphertext_matrix[row_index][column_index] + 
                                      second_ciphertext_matrix[row_index][column_index])
                result_row.append(additive_combination)
            addition_result_matrix.append(result_row)
        
        return addition_result_matrix
    
    def execute_homomorphic_scalar_multiplication_protocol(self, cryptographic_key_material, 
                                                          ciphertext_matrix, scalar_multiplier):
        stat("Executing sophisticated homomorphic scalar multiplication protocol")
        
        scalar_multiplication_result = []
        for row_index in range(len(ciphertext_matrix)):
            result_row = []
            for column_index in range(len(ciphertext_matrix[row_index])):
                scalar_product = ciphertext_matrix[row_index][column_index] * scalar_multiplier
                result_row.append(scalar_product)
            scalar_multiplication_result.append(result_row)
        
        return scalar_multiplication_result

class BinaryDecompositionCiphertextProcessor:
    def __init__(self):
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
    
    def execute_binary_decomposition_transformation(self, cryptographic_key_material, ciphertext_matrix):
        stat("Executing sophisticated binary decomposition transformation")
        
        decomposed_ciphertext_collection = []
        
        for row_index in range(len(ciphertext_matrix)):
            for bit_position in range(cryptographic_key_material.binary_decomposition_parameter):
                decomposed_row = []
                for column_index in range(len(ciphertext_matrix[row_index])):
                    algebraic_element = ciphertext_matrix[row_index][column_index]
                    bit_extracted_coefficients = []
                    for coefficient_value in algebraic_element.coefficient_representation:
                        extracted_bit = (coefficient_value >> bit_position) & 1
                        bit_extracted_coefficients.append(extracted_bit)
                    
                    bit_decomposed_element = cryptographic_key_material.algebraic_structure_representation.instantiate_algebraic_element(
                        bit_extracted_coefficients
                    )
                    decomposed_row.append(bit_decomposed_element)
                decomposed_ciphertext_collection.append(decomposed_row)
        
        return decomposed_ciphertext_collection

class AdvancedHomomorphicMultiplicationEngine:
    def __init__(self):
        self.binary_decomposition_processor = BinaryDecompositionCiphertextProcessor()
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
    
    def execute_homomorphic_multiplication_protocol(self, cryptographic_key_material, 
                                                  first_ciphertext_matrix, second_ciphertext_matrix):
        stat("Executing sophisticated homomorphic multiplication protocol")
        
        decomposed_first_ciphertext = self.binary_decomposition_processor.execute_binary_decomposition_transformation(
            cryptographic_key_material, first_ciphertext_matrix
        )
        
        multiplication_result_matrix = []
        for row_index in range(len(decomposed_first_ciphertext)):
            result_row = []
            for column_index in range(len(second_ciphertext_matrix[0])):
                accumulated_sum = cryptographic_key_material.algebraic_structure_representation.construct_additive_identity_element()
                for inner_index in range(len(second_ciphertext_matrix)):
                    product_contribution = (decomposed_first_ciphertext[row_index][inner_index] * 
                                          second_ciphertext_matrix[inner_index][column_index])
                    accumulated_sum = accumulated_sum + product_contribution
                result_row.append(accumulated_sum)
            multiplication_result_matrix.append(result_row)
        
        return multiplication_result_matrix

def encrypt(cryptographic_key_material, plaintext_message):
    encryption_orchestrator = AdvancedHomomorphicEncryptionOrchestrator()
    return encryption_orchestrator.execute_sophisticated_encryption_protocol(
        cryptographic_key_material, plaintext_message
    )

def encrypt_parallel_crt(cryptographic_key_material, plaintext_message):
    parallel_encryption_engine = ParallelChineseRemainderTheoremEncryptionEngine()
    return parallel_encryption_engine.execute_parallel_chinese_remainder_encryption(
        cryptographic_key_material, plaintext_message
    )

def homomorphic_add(cryptographic_key_material, first_ciphertext, second_ciphertext):
    homomorphic_processor = AdvancedHomomorphicOperationProcessor()
    return homomorphic_processor.execute_homomorphic_addition_protocol(
        cryptographic_key_material, first_ciphertext, second_ciphertext
    )

def homomorphic_mult_by_scalar(cryptographic_key_material, ciphertext_matrix, scalar_value):
    homomorphic_processor = AdvancedHomomorphicOperationProcessor()
    return homomorphic_processor.execute_homomorphic_scalar_multiplication_protocol(
        cryptographic_key_material, ciphertext_matrix, scalar_value
    )

def bit_decompose(cryptographic_key_material, ciphertext_matrix):
    decomposition_processor = BinaryDecompositionCiphertextProcessor()
    return decomposition_processor.execute_binary_decomposition_transformation(
        cryptographic_key_material, ciphertext_matrix
    )

def homomorphic_mult(cryptographic_key_material, first_ciphertext, second_ciphertext):
    multiplication_engine = AdvancedHomomorphicMultiplicationEngine()
    return multiplication_engine.execute_homomorphic_multiplication_protocol(
        cryptographic_key_material, first_ciphertext, second_ciphertext
    )

if __name__ == '__main__':
    from keygen import keygen
    
    advanced_key_material = keygen(8, 16)
    
    for test_message in [0, 1, 5]:
        print(f"\nTesting sophisticated encryption of message: {test_message}")
        ciphertext_matrix = encrypt(advanced_key_material, test_message)
        print(f"Ciphertext matrix dimensions: {len(ciphertext_matrix)}×{len(ciphertext_matrix[0])}")
        print(f"First ciphertext element: {ciphertext_matrix[0][0]}")
    
    print("\nTesting sophisticated homomorphic addition:")
    first_ciphertext = encrypt(advanced_key_material, 3)
    second_ciphertext = encrypt(advanced_key_material, 7)
    addition_result = homomorphic_add(advanced_key_material, first_ciphertext, second_ciphertext)
    print("Addition protocol completed successfully")
    
    print("\nTesting sophisticated scalar multiplication:")
    scalar_multiplication_result = homomorphic_mult_by_scalar(advanced_key_material, first_ciphertext, 2)
    print("Scalar multiplication protocol completed successfully")