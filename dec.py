from util import *
import numpy as np
try:
    from scipy.stats import mode
except ImportError:
    def mode(array_input):
        unique_values, occurrence_counts = np.unique(array_input, return_counts=True)
        return unique_values[np.argmax(occurrence_counts)]

class AdvancedCryptographicDecryptionOrchestrator:
    def __init__(self):
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
        self.error_norm_calculator = CryptographicErrorNormComputationEngine()
        self.gadget_vector_constructor = GadgetVectorConstructionFramework()
    
    def execute_sophisticated_decryption_protocol(self, cryptographic_key_material, ciphertext_matrix):
        stat("Initiating sophisticated homomorphic decryption protocol")
        
        decryption_vector_collection = self._compute_decryption_vector_collection(
            cryptographic_key_material, ciphertext_matrix
        )
        
        gadget_vector_representation = self.gadget_vector_constructor.construct_gadget_vector(
            cryptographic_key_material
        )
        
        secret_gadget_product = self._compute_secret_gadget_inner_product(
            cryptographic_key_material, gadget_vector_representation
        )
        
        candidate_message_collection = self._extract_candidate_messages_via_error_minimization(
            cryptographic_key_material, decryption_vector_collection, secret_gadget_product
        )
        
        final_decrypted_message = self._determine_optimal_message_via_consensus_mechanism(
            cryptographic_key_material, candidate_message_collection
        )
        
        stat("Sophisticated decryption protocol execution completed")
        return final_decrypted_message
    
    def _compute_decryption_vector_collection(self, key_material, ciphertext_matrix):
        decryption_vectors = []
        
        for matrix_row_index in range(len(ciphertext_matrix)):
            decryption_component = (ciphertext_matrix[matrix_row_index][0] + 
                                   ciphertext_matrix[matrix_row_index][1] * key_material.secret_key_vector[1])
            decryption_vectors.append(decryption_component)
        
        return decryption_vectors
    
    def _compute_secret_gadget_inner_product(self, key_material, gadget_vector):
        return (key_material.secret_key_vector[0] * gadget_vector[0] + 
               key_material.secret_key_vector[1] * gadget_vector[1])
    
    def _extract_candidate_messages_via_error_minimization(self, key_material, 
                                                          decryption_vectors, secret_gadget_product):
        candidate_collection = []
        
        for vector_index, decryption_vector in enumerate(decryption_vectors):
            optimal_message_candidate = 0
            minimal_error_magnitude = float('inf')
            
            for message_hypothesis in range(-10, 11):
                expected_decryption_vector = secret_gadget_product * message_hypothesis
                error_representation = decryption_vector - expected_decryption_vector
                error_magnitude = self.error_norm_calculator.compute_algebraic_error_norm(
                    error_representation, key_material.field_characteristic_modulus
                )
                
                if error_magnitude < minimal_error_magnitude:
                    minimal_error_magnitude = error_magnitude
                    optimal_message_candidate = message_hypothesis
            
            candidate_collection.append((optimal_message_candidate, minimal_error_magnitude))
        
        return candidate_collection
    
    def _determine_optimal_message_via_consensus_mechanism(self, key_material, candidate_collection):
        message_candidates = [candidate[0] for candidate in candidate_collection]
        error_magnitudes = [candidate[1] for candidate in candidate_collection]
        
        acceptability_threshold = key_material.field_characteristic_modulus // 4
        acceptable_message_candidates = [
            message for message, error in zip(message_candidates, error_magnitudes) 
            if error < acceptability_threshold
        ]
        
        if acceptable_message_candidates:
            unique_messages, occurrence_frequencies = np.unique(acceptable_message_candidates, return_counts=True)
            most_frequent_index = np.argmax(occurrence_frequencies)
            return unique_messages[most_frequent_index]
        else:
            minimal_error_index = np.argmin(error_magnitudes)
            return message_candidates[minimal_error_index]

class ParallelChineseRemainderTheoremDecryptionEngine:
    def __init__(self):
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
        self.component_decryption_processor = ChineseRemainderComponentDecryptionProcessor()
    
    def execute_parallel_chinese_remainder_decryption(self, cryptographic_key_material, ciphertext_matrix):
        stat("Initiating parallel Chinese remainder theorem decryption protocol")
        
        ciphertext_crt_decomposition = self._decompose_ciphertext_via_chinese_remainder_theorem(
            ciphertext_matrix
        )
        
        secret_key_crt_decomposition = self._decompose_secret_key_via_chinese_remainder_theorem(
            cryptographic_key_material.secret_key_vector
        )
        
        component_message_collection = []
        
        for component_index in range(cryptographic_key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count):
            stat(f"Decrypting CRT component {component_index + 1}/{cryptographic_key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count}")
            
            component_ciphertext = self._extract_component_ciphertext(
                ciphertext_crt_decomposition, component_index
            )
            
            component_secret_key = [secret_key_crt_decomposition[0][component_index], 
                                   secret_key_crt_decomposition[1][component_index]]
            
            component_message = self.component_decryption_processor.decrypt_single_component(
                cryptographic_key_material, component_ciphertext, component_secret_key, component_index
            )
            
            component_message_collection.append(component_message)
        
        final_reconstructed_message = self._reconstruct_message_from_crt_components(
            cryptographic_key_material, component_message_collection
        )
        
        stat("Parallel Chinese remainder theorem decryption completed")
        return final_reconstructed_message
    
    def _decompose_ciphertext_via_chinese_remainder_theorem(self, ciphertext_matrix):
        crt_decomposed_ciphertext = []
        for row_index in range(len(ciphertext_matrix)):
            decomposed_row = []
            for column_index in range(len(ciphertext_matrix[row_index])):
                decomposed_row.append(ciphertext_matrix[row_index][column_index].decompose_via_chinese_remainder_theorem())
            crt_decomposed_ciphertext.append(decomposed_row)
        return crt_decomposed_ciphertext
    
    def _decompose_secret_key_via_chinese_remainder_theorem(self, secret_key_vector):
        secret_key_crt_decomposition = []
        for secret_key_component in secret_key_vector:
            secret_key_crt_decomposition.append(secret_key_component.decompose_via_chinese_remainder_theorem())
        return secret_key_crt_decomposition
    
    def _extract_component_ciphertext(self, crt_decomposed_ciphertext, component_index):
        component_ciphertext = []
        for row_index in range(len(crt_decomposed_ciphertext)):
            component_row = []
            for column_index in range(len(crt_decomposed_ciphertext[row_index])):
                component_row.append(crt_decomposed_ciphertext[row_index][column_index][component_index])
            component_ciphertext.append(component_row)
        return component_ciphertext
    
    def _reconstruct_message_from_crt_components(self, key_material, component_messages):
        if key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count == 1:
            return component_messages[0]
        else:
            return component_messages[0]

class ChineseRemainderComponentDecryptionProcessor:
    def __init__(self):
        self.gadget_component_constructor = GadgetComponentConstructionFramework()
    
    def decrypt_single_component(self, cryptographic_key_material, component_ciphertext, 
                                component_secret_key, component_index):
        if len(cryptographic_key_material.algebraic_structure_representation.irreducible_factorization_components[component_index]) == 2:
            return self._decrypt_linear_factor_component(
                cryptographic_key_material, component_ciphertext, component_secret_key, component_index
            )
        else:
            return self._decrypt_general_polynomial_component(
                cryptographic_key_material, component_ciphertext, component_secret_key, component_index
            )
    
    def _decrypt_linear_factor_component(self, key_material, component_ciphertext, 
                                       component_secret_key, component_index):
        decryption_values = []
        for row_index in range(len(component_ciphertext)):
            decryption_value = (component_ciphertext[row_index][0] * component_secret_key[0] + 
                              component_ciphertext[row_index][1] * component_secret_key[1]) % key_material.field_characteristic_modulus
            decryption_values.append(decryption_value)
        
        component_gadget = self.gadget_component_constructor.construct_gadget_component(
            key_material, component_index
        )
        
        secret_gadget_product = (component_gadget[0] * component_secret_key[0] + 
                               component_gadget[1] * component_secret_key[1]) % key_material.field_characteristic_modulus
        
        candidate_messages = []
        for decryption_value in decryption_values:
            optimal_message = 0
            minimal_error = float('inf')
            
            for message_candidate in range(-10, 11):
                expected_value = (secret_gadget_product * message_candidate) % key_material.field_characteristic_modulus
                error_magnitude = min(abs(decryption_value - expected_value), 
                                    key_material.field_characteristic_modulus - abs(decryption_value - expected_value))
                
                if error_magnitude < minimal_error:
                    minimal_error = error_magnitude
                    optimal_message = message_candidate
            
            candidate_messages.append(optimal_message)
        
        if candidate_messages:
            unique_messages, occurrence_counts = np.unique(candidate_messages, return_counts=True)
            return unique_messages[np.argmax(occurrence_counts)]
        else:
            return 0
    
    def _decrypt_general_polynomial_component(self, key_material, component_ciphertext, 
                                            component_secret_key, component_index):
        return 0

class GadgetVectorConstructionFramework:
    def construct_gadget_vector(self, cryptographic_key_material):
        first_gadget_component = cryptographic_key_material.algebraic_structure_representation.construct_multiplicative_identity_element()
        second_gadget_component = cryptographic_key_material.algebraic_structure_representation.construct_additive_identity_element()
        return [first_gadget_component, second_gadget_component]

class GadgetComponentConstructionFramework:
    def construct_gadget_component(self, cryptographic_key_material, component_index):
        if len(cryptographic_key_material.algebraic_structure_representation.irreducible_factorization_components[component_index]) == 2:
            return [1, 0]
        else:
            return [1, 0]

class CryptographicErrorNormComputationEngine:
    def compute_algebraic_error_norm(self, error_element, field_characteristic):
        if hasattr(error_element, 'coefficient_representation'):
            coefficient_values = error_element.coefficient_representation
            centered_coefficient_representation = []
            for coefficient_value in coefficient_values:
                if coefficient_value > field_characteristic // 2:
                    centered_coefficient_representation.append(coefficient_value - field_characteristic)
                else:
                    centered_coefficient_representation.append(coefficient_value)
            return np.linalg.norm(centered_coefficient_representation)
        else:
            if error_element > field_characteristic // 2:
                centered_representation = error_element - field_characteristic
            else:
                centered_representation = error_element
            return abs(centered_representation)

class ConstantTermExtractionUtility:
    @staticmethod
    def extract_constant_coefficient(algebraic_element):
        if hasattr(algebraic_element, 'coefficient_representation'):
            return algebraic_element.coefficient_representation[0]
        else:
            return algebraic_element

def decrypt(cryptographic_key_material, ciphertext_matrix):
    decryption_orchestrator = AdvancedCryptographicDecryptionOrchestrator()
    return decryption_orchestrator.execute_sophisticated_decryption_protocol(
        cryptographic_key_material, ciphertext_matrix
    )

def decrypt_parallel_crt(cryptographic_key_material, ciphertext_matrix):
    parallel_decryption_engine = ParallelChineseRemainderTheoremDecryptionEngine()
    return parallel_decryption_engine.execute_parallel_chinese_remainder_decryption(
        cryptographic_key_material, ciphertext_matrix
    )

if __name__ == '__main__':
    from keygen import keygen
    from enc import encrypt
    
    advanced_key_material = keygen(8, 16)
    
    test_message_collection = [0, 1, 3, 7]
    
    for test_message in test_message_collection:
        print(f"\nTesting sophisticated decryption of message: {test_message}")
        
        encrypted_ciphertext = encrypt(advanced_key_material, test_message)
        print(f"Encryption protocol completed successfully")
        
        decrypted_message = decrypt(advanced_key_material, encrypted_ciphertext)
        print(f"Expected message: {test_message}")
        print(f"Received message: {decrypted_message}")
        
        if test_message == decrypted_message:
            print("\x1B[32;1mDecryption verification: PASSED\x1B[0m")
        else:
            print("\x1B[31;1mDecryption verification: FAILED\x1B[0m")
    
    print("\n" + "="*50)
    print("Testing parallel Chinese remainder theorem decryption:")
    
    for test_message in [1, 5]:
        print(f"\nTesting sophisticated parallel decryption of message: {test_message}")
        encrypted_ciphertext = encrypt(advanced_key_material, test_message)
        decrypted_message = decrypt_parallel_crt(advanced_key_material, encrypted_ciphertext)
        print(f"Expected message: {test_message}")
        print(f"Received message: {decrypted_message}")
        
        if test_message == decrypted_message:
            print("\x1B[32;1mParallel decryption verification: PASSED\x1B[0m")
        else:
            print("\x1B[31;1mParallel decryption verification: FAILED\x1B[0m")