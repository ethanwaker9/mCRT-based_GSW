import numpy as np
from util import stat, AlgebraicElementRepresentation
from keygen import keygen
from enc import encrypt, homomorphic_add, homomorphic_mult_by_scalar
from dec import decrypt

class ComprehensiveCryptographicTestingFramework:
    def __init__(self):
        self.performance_monitor = TemporalPerformanceMetricsAggregator()
        self.test_validation_registry = {}
    
    def execute_fundamental_encryption_decryption_verification_protocol(self):
        print("="*60)
        print("Executing Fundamental mCRT-GSW Encryption/Decryption Verification")
        print("="*60)
        
        cryptographic_key_material = keygen(8, 20)
        
        verification_message_collection = [0, 1, 2, 5, 7]
        
        for verification_message in verification_message_collection:
            print(f"\nProcessing verification message: {verification_message}")
            
            encrypted_ciphertext_matrix = encrypt(cryptographic_key_material, verification_message)
            decrypted_message_result = decrypt(cryptographic_key_material, encrypted_ciphertext_matrix)
            
            print(" "*12 + "Expected result: %d" % verification_message)
            print(" "*12 + "Received result: %d" % decrypted_message_result)
            
            if verification_message == decrypted_message_result:
                print(" "*12 + "\x1B[32;1mVerification Status: PASSED\x1B[0m")
                self.test_validation_registry[f"basic_encryption_{verification_message}"] = True
            else:
                print(" "*12 + "\x1B[31;1mVerification Status: FAILED\x1B[0m")
                self.test_validation_registry[f"basic_encryption_{verification_message}"] = False
    
    def execute_homomorphic_addition_verification_protocol(self):
        print("\n" + "="*60)
        print("Executing Homomorphic Addition Verification Protocol")
        print("="*60)
        
        cryptographic_key_material = keygen(8, 20)
        
        addition_test_pair_collection = [(1, 1), (3, 4), (2, 5), (0, 7)]
        
        for first_operand, second_operand in addition_test_pair_collection:
            print(f"\nProcessing addition verification: {first_operand} + {second_operand}")
            
            first_encrypted_ciphertext = encrypt(cryptographic_key_material, first_operand)
            second_encrypted_ciphertext = encrypt(cryptographic_key_material, second_operand)
            
            homomorphic_addition_result = homomorphic_add(
                cryptographic_key_material, first_encrypted_ciphertext, second_encrypted_ciphertext
            )
            
            decrypted_addition_result = decrypt(cryptographic_key_material, homomorphic_addition_result)
            expected_addition_result = first_operand + second_operand
            
            print(" "*12 + "Expected result: %d" % expected_addition_result)
            print(" "*12 + "Received result: %d" % decrypted_addition_result)
            
            if expected_addition_result == decrypted_addition_result:
                print(" "*12 + "\x1B[32;1mAddition Verification: PASSED\x1B[0m")
                self.test_validation_registry[f"homomorphic_add_{first_operand}_{second_operand}"] = True
            else:
                print(" "*12 + "\x1B[31;1mAddition Verification: FAILED\x1B[0m")
                self.test_validation_registry[f"homomorphic_add_{first_operand}_{second_operand}"] = False
    
    def execute_scalar_multiplication_verification_protocol(self):
        print("\n" + "="*60)
        print("Executing Homomorphic Scalar Multiplication Verification Protocol")
        print("="*60)
        
        cryptographic_key_material = keygen(8, 20)
        
        scalar_multiplication_test_cases = [(3, 2), (5, 3), (1, 7), (4, 0)]
        
        for plaintext_message, scalar_multiplier in scalar_multiplication_test_cases:
            print(f"\nProcessing scalar multiplication verification: {plaintext_message} * {scalar_multiplier}")
            
            encrypted_ciphertext = encrypt(cryptographic_key_material, plaintext_message)
            
            homomorphic_scalar_multiplication_result = homomorphic_mult_by_scalar(
                cryptographic_key_material, encrypted_ciphertext, scalar_multiplier
            )
            
            decrypted_multiplication_result = decrypt(cryptographic_key_material, homomorphic_scalar_multiplication_result)
            expected_multiplication_result = plaintext_message * scalar_multiplier
            
            print(" "*12 + "Expected result: %d" % expected_multiplication_result)
            print(" "*12 + "Received result: %d" % decrypted_multiplication_result)
            
            if expected_multiplication_result == decrypted_multiplication_result:
                print(" "*12 + "\x1B[32;1mScalar Multiplication Verification: PASSED\x1B[0m")
                self.test_validation_registry[f"scalar_mult_{plaintext_message}_{scalar_multiplier}"] = True
            else:
                print(" "*12 + "\x1B[31;1mScalar Multiplication Verification: FAILED\x1B[0m")
                self.test_validation_registry[f"scalar_mult_{plaintext_message}_{scalar_multiplier}"] = False
    
    def execute_complex_homomorphic_operations_verification_protocol(self):
        print("\n" + "="*60)
        print("Executing Complex Homomorphic Operations Verification Protocol")
        print("="*60)
        
        cryptographic_key_material = keygen(8, 20)
        
        first_operand, second_operand = 4, 2
        print(f"\nProcessing complex operation verification: 2*{first_operand} + 3*{second_operand}")
        
        first_encrypted_ciphertext = encrypt(cryptographic_key_material, first_operand)
        second_encrypted_ciphertext = encrypt(cryptographic_key_material, second_operand)
        
        first_scaled_ciphertext = homomorphic_mult_by_scalar(cryptographic_key_material, first_encrypted_ciphertext, 2)
        second_scaled_ciphertext = homomorphic_mult_by_scalar(cryptographic_key_material, second_encrypted_ciphertext, 3)
        complex_operation_result = homomorphic_add(cryptographic_key_material, first_scaled_ciphertext, second_scaled_ciphertext)
        
        decrypted_complex_result = decrypt(cryptographic_key_material, complex_operation_result)
        expected_complex_result = 2*first_operand + 3*second_operand
        
        print(" "*12 + "Expected result: %d" % expected_complex_result)
        print(" "*12 + "Received result: %d" % decrypted_complex_result)
        
        if expected_complex_result == decrypted_complex_result:
            print(" "*12 + "\x1B[32;1mComplex Operation Verification: PASSED\x1B[0m")
            self.test_validation_registry["complex_operation_1"] = True
        else:
            print(" "*12 + "\x1B[31;1mComplex Operation Verification: FAILED\x1B[0m")
            self.test_validation_registry["complex_operation_1"] = False
        
        print(f"\nProcessing multiple addition verification: {first_operand} + {first_operand} + {first_operand} + {second_operand} + {second_operand} + {second_operand}")
        
        first_accumulated_sum = homomorphic_add(cryptographic_key_material, first_encrypted_ciphertext, first_encrypted_ciphertext)
        first_accumulated_sum = homomorphic_add(cryptographic_key_material, first_accumulated_sum, first_encrypted_ciphertext)
        second_accumulated_sum = homomorphic_add(cryptographic_key_material, second_encrypted_ciphertext, second_encrypted_ciphertext)
        second_accumulated_sum = homomorphic_add(cryptographic_key_material, second_accumulated_sum, second_encrypted_ciphertext)
        final_accumulated_sum = homomorphic_add(cryptographic_key_material, first_accumulated_sum, second_accumulated_sum)
        
        decrypted_final_result = decrypt(cryptographic_key_material, final_accumulated_sum)
        expected_final_result = 3*first_operand + 3*second_operand
        
        print(" "*12 + "Expected result: %d" % expected_final_result)
        print(" "*12 + "Received result: %d" % decrypted_final_result)
        
        if expected_final_result == decrypted_final_result:
            print(" "*12 + "\x1B[32;1mMultiple Addition Verification: PASSED\x1B[0m")
            self.test_validation_registry["complex_operation_2"] = True
        else:
            print(" "*12 + "\x1B[31;1mMultiple Addition Verification: FAILED\x1B[0m")
            self.test_validation_registry["complex_operation_2"] = False
    
    def execute_algebraic_structure_properties_verification_protocol(self):
        print("\n" + "="*60)
        print("Executing Algebraic Structure Properties Verification Protocol")
        print("="*60)
        
        cryptographic_key_material = keygen(4, 16)
        
        print("\nAlgebraic structure configuration:")
        print(f"  Ring dimension parameter: {cryptographic_key_material.ring_dimension_parameter}")
        print(f"  Field characteristic modulus: {cryptographic_key_material.field_characteristic_modulus}")
        print(f"  CRT component count: {cryptographic_key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count}")
        print(f"  Extended matrix dimension: {cryptographic_key_material.extended_matrix_dimension}×2")
        
        print("\nAlgebraic element operation verification:")
        
        first_random_element = cryptographic_key_material.algebraic_structure_representation.generate_uniformly_distributed_element()
        second_random_element = cryptographic_key_material.algebraic_structure_representation.generate_uniformly_distributed_element()
        
        print(f"  r1 = {first_random_element}")
        print(f"  r2 = {second_random_element}")
        print(f"  r1 + r2 = {first_random_element + second_random_element}")
        print(f"  r1 * r2 = {first_random_element * second_random_element}")
        
        print("\nChinese remainder theorem decomposition verification:")
        crt_decomposition_first = first_random_element.decompose_via_chinese_remainder_theorem()
        print(f"  CRT components of r1: {crt_decomposition_first}")
        
        print("\nAlgebraic element encryption verification:")
        encrypted_algebraic_element = encrypt(cryptographic_key_material, first_random_element)
        decrypted_algebraic_element = decrypt(cryptographic_key_material, encrypted_algebraic_element)
        
        expected_constant_term = first_random_element.coefficient_representation[0] if hasattr(first_random_element, 'coefficient_representation') else first_random_element
        
        print(f"  Expected constant term: {expected_constant_term}")
        print(f"  Received result: {decrypted_algebraic_element}")
    
    def execute_error_resilience_verification_protocol(self):
        print("\n" + "="*60)
        print("Executing Error Resilience Verification Protocol")
        print("="*60)
        
        cryptographic_key_material = keygen(8, 20)
        
        print("\nZero value encryption verification:")
        zero_encrypted_ciphertext = encrypt(cryptographic_key_material, 0)
        zero_decrypted_result = decrypt(cryptographic_key_material, zero_encrypted_ciphertext)
        print(f"  Expected: 0, Received: {zero_decrypted_result}")
        
        print("\nLarge value encryption verification:")
        for large_test_message in [10, 15, 20]:
            large_encrypted_ciphertext = encrypt(cryptographic_key_material, large_test_message)
            large_decrypted_result = decrypt(cryptographic_key_material, large_encrypted_ciphertext)
            print(f"  Message {large_test_message}: Expected {large_test_message}, Received {large_decrypted_result}")
            if large_test_message == large_decrypted_result:
                print("    \x1B[32;1mLarge Value Verification: PASSED\x1B[0m")
                self.test_validation_registry[f"error_resilience_{large_test_message}"] = True
            else:
                print("    \x1B[31;1mLarge Value Verification: FAILED\x1B[0m")
                self.test_validation_registry[f"error_resilience_{large_test_message}"] = False
    
    def demonstrate_parallel_chinese_remainder_theorem_computation(self):
        print("\n" + "="*60)
        print("Demonstrating Parallel Chinese Remainder Theorem Computation")
        print("="*60)
        
        cryptographic_key_material = keygen(8, 20)
        
        print(f"Algebraic structure contains {cryptographic_key_material.algebraic_structure_representation.chinese_remainder_theorem_component_count} CRT components")
        print("Computational operations can be parallelized across these components")
        
        test_algebraic_element = cryptographic_key_material.algebraic_structure_representation.instantiate_algebraic_element([5, 3, 1, 0, 0, 0, 0, 0])
        print(f"\nTest algebraic element: {test_algebraic_element}")
        
        crt_decomposition_components = test_algebraic_element.decompose_via_chinese_remainder_theorem()
        print(f"CRT decomposition: {crt_decomposition_components}")
        
        print("\nExecuting encryption with CRT decomposition awareness...")
        encrypted_ciphertext = encrypt(cryptographic_key_material, 7)
        decrypted_result = decrypt(cryptographic_key_material, encrypted_ciphertext)
        
        print(f"Plaintext message: 7")
        print(f"Decrypted result: {decrypted_result}")
    
    def generate_comprehensive_test_report(self):
        print("\n" + "="*60)
        print("Comprehensive Test Execution Report")
        print("="*60)
        
        total_tests = len(self.test_validation_registry)
        passed_tests = sum(self.test_validation_registry.values())
        failed_tests = total_tests - passed_tests
        
        print(f"\nTotal test cases executed: {total_tests}")
        print(f"Successful verifications: {passed_tests}")
        print(f"Failed verifications: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "Success rate: N/A")
        
        if failed_tests > 0:
            print("\nFailed test cases:")
            for test_name, result in self.test_validation_registry.items():
                if not result:
                    print(f"  - {test_name}")

def test_basic_encryption():
    testing_framework = ComprehensiveCryptographicTestingFramework()
    testing_framework.execute_fundamental_encryption_decryption_verification_protocol()

def test_homomorphic_addition():
    testing_framework = ComprehensiveCryptographicTestingFramework()
    testing_framework.execute_homomorphic_addition_verification_protocol()

def test_scalar_multiplication():
    testing_framework = ComprehensiveCryptographicTestingFramework()
    testing_framework.execute_scalar_multiplication_verification_protocol()

def test_complex_operations():
    testing_framework = ComprehensiveCryptographicTestingFramework()
    testing_framework.execute_complex_homomorphic_operations_verification_protocol()

def test_ring_properties():
    testing_framework = ComprehensiveCryptographicTestingFramework()
    testing_framework.execute_algebraic_structure_properties_verification_protocol()

def test_error_resilience():
    testing_framework = ComprehensiveCryptographicTestingFramework()
    testing_framework.execute_error_resilience_verification_protocol()

def demonstrate_parallel_crt():
    testing_framework = ComprehensiveCryptographicTestingFramework()
    testing_framework.demonstrate_parallel_chinese_remainder_theorem_computation()

if __name__ == '__main__':
    print("Advanced mCRT-RLWE based GSW Homomorphic Encryption System")
    print("Sophisticated Implementation for Parallel Computation and CRT Optimization")
    print()
    
    comprehensive_testing_framework = ComprehensiveCryptographicTestingFramework()
    
    comprehensive_testing_framework.execute_fundamental_encryption_decryption_verification_protocol()
    comprehensive_testing_framework.execute_homomorphic_addition_verification_protocol()
    comprehensive_testing_framework.execute_scalar_multiplication_verification_protocol()
    comprehensive_testing_framework.execute_complex_homomorphic_operations_verification_protocol()
    comprehensive_testing_framework.execute_algebraic_structure_properties_verification_protocol()
    comprehensive_testing_framework.execute_error_resilience_verification_protocol()
    comprehensive_testing_framework.demonstrate_parallel_chinese_remainder_theorem_computation()
    
    comprehensive_testing_framework.generate_comprehensive_test_report()
    
    print("\n" + "="*60)
    print("Advanced Testing Protocol Execution Completed Successfully!")
    print("="*60)
    