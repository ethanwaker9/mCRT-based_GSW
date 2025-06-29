import time
from math import floor, ceil, log2
from random import randint
import numpy as np
from numpy.polynomial import polynomial as P

class TemporalPerformanceMetricsAggregator:
    def __init__(self):
        self.initialization_timestamp = None
        self.performance_measurement_registry = {}
    
    def register_computational_checkpoint(self, operational_descriptor):
        current_temporal_coordinate = time.time()
        if self.initialization_timestamp is None:
            self.initialization_timestamp = current_temporal_coordinate
        elapsed_duration = current_temporal_coordinate - self.initialization_timestamp
        formatted_output = "\x1B[2m%10.4f  %s\x1B[0m" % (elapsed_duration, operational_descriptor)
        print(formatted_output)
        self.performance_measurement_registry[operational_descriptor] = elapsed_duration

class ModularArithmeticComputationalEngine:
    @staticmethod
    def compute_modular_exponentiation_via_binary_method(base_operand, exponent_parameter, modular_parameter):
        if modular_parameter == 1:
            return 0
        if exponent_parameter == 0:
            return 1
        
        accumulator_register = 1
        normalized_base = base_operand % modular_parameter
        
        while exponent_parameter > 0:
            if exponent_parameter % 2 == 1:
                accumulator_register = (accumulator_register * normalized_base) % modular_parameter
            exponent_parameter = exponent_parameter >> 1
            normalized_base = (normalized_base * normalized_base) % modular_parameter
        
        return accumulator_register

class ProbabilisticPrimalityTestingFramework:
    def __init__(self, precision_threshold=20):
        self.precision_threshold = precision_threshold
        self.arithmetic_engine = ModularArithmeticComputationalEngine()
    
    def execute_miller_rabin_compositeness_verification(self, candidate_prime):
        if candidate_prime < 2:
            return False
        if candidate_prime == 2:
            return True
        if candidate_prime % 2 == 0:
            return False
        
        iteration_count = self.precision_threshold if candidate_prime.bit_length() > 64 else 16
        
        for _ in range(iteration_count):
            witness_value = randint(2, candidate_prime - 2)
            if self.arithmetic_engine.compute_modular_exponentiation_via_binary_method(
                witness_value, candidate_prime - 1, candidate_prime) != 1:
                return False
        return True

class CryptographicPrimeGenerationOracle:
    def __init__(self):
        self.primality_tester = ProbabilisticPrimalityTestingFramework()
        self.maximum_generation_attempts = 1000
    
    def synthesize_prime_with_specified_bit_length(self, target_bit_length):
        if target_bit_length < 2:
            return 2
        
        attempt_counter = 0
        
        while attempt_counter < self.maximum_generation_attempts:
            candidate_value = randint(2**(target_bit_length-1), 2**target_bit_length - 1)
            if candidate_value % 2 == 0:
                candidate_value += 1
            
            if self.primality_tester.execute_miller_rabin_compositeness_verification(candidate_value):
                return candidate_value
            
            attempt_counter += 1
        
        fallback_candidate = 2**(target_bit_length-1) + 1
        while fallback_candidate < 2**target_bit_length:
            if self.primality_tester.execute_miller_rabin_compositeness_verification(fallback_candidate):
                return fallback_candidate
            fallback_candidate += 2
        
        raise RuntimeError(f"Prime generation failure for {target_bit_length} bit specification")
    
    def generate_sophie_germain_prime_configuration(self, target_bit_specification):
        generation_attempt_limit = 200
        current_attempt = 0
        
        while current_attempt < generation_attempt_limit:
            base_prime_candidate = self.synthesize_prime_with_specified_bit_length(target_bit_specification - 1)
            safe_prime_candidate = 2 * base_prime_candidate + 1
            if self.primality_tester.execute_miller_rabin_compositeness_verification(safe_prime_candidate):
                return base_prime_candidate
            current_attempt += 1
        
        return self.synthesize_prime_with_specified_bit_length(target_bit_specification)

class AbstractAlgebraicStructureRepresentation:
    def __init__(self, field_characteristic, polynomial_descriptor, factorization_decomposition=None):
        self.field_characteristic = field_characteristic
        self.polynomial_representation_vector = self._normalize_polynomial_representation(polynomial_descriptor)
        self.algebraic_degree = len(self.polynomial_representation_vector) - 1
        self.irreducible_factorization_components = (
            factorization_decomposition if factorization_decomposition is not None 
            else self._execute_factorization_decomposition()
        )
        self.chinese_remainder_theorem_component_count = len(self.irreducible_factorization_components)
    
    def _normalize_polynomial_representation(self, polynomial_descriptor):
        if isinstance(self.field_characteristic, int) and self.field_characteristic.bit_length() > 63:
            if isinstance(polynomial_descriptor, list):
                return np.array(polynomial_descriptor, dtype=object)
            else:
                return polynomial_descriptor.astype(object)
        else:
            if isinstance(polynomial_descriptor, list):
                return np.array(polynomial_descriptor, dtype=np.int64)
            else:
                return polynomial_descriptor.astype(np.int64)
    
    def _execute_factorization_decomposition(self):
        factorization_registry = []
        if self.algebraic_degree == 2:
            root_extraction_limit = min(self.field_characteristic, 1000)
            for potential_root in range(root_extraction_limit):
                if (potential_root * potential_root + 1) % self.field_characteristic == 0:
                    factorization_registry.append([1, -potential_root])
        
        if len(factorization_registry) == 0:
            factorization_registry = [self.polynomial_representation_vector.copy()]
        
        return factorization_registry
    
    def instantiate_algebraic_element(self, coefficient_specification):
        return AlgebraicElementRepresentation(self, coefficient_specification)
    
    def generate_uniformly_distributed_element(self):
        if self.field_characteristic.bit_length() > 63:
            coefficient_array = []
            for _ in range(self.algebraic_degree):
                maximum_coefficient_magnitude = min(self.field_characteristic, 2**32)
                random_coefficient = np.random.randint(0, maximum_coefficient_magnitude)
                coefficient_array.append(random_coefficient)
            return self.instantiate_algebraic_element(coefficient_array)
        else:
            coefficient_array = np.random.randint(
                0, self.field_characteristic, 
                size=self.algebraic_degree, 
                dtype=np.int64
            )
            return self.instantiate_algebraic_element(coefficient_array)
    
    def construct_additive_identity_element(self):
        if self.field_characteristic.bit_length() > 63:
            return self.instantiate_algebraic_element([0] * self.algebraic_degree)
        else:
            return self.instantiate_algebraic_element(np.zeros(self.algebraic_degree, dtype=np.int64))
    
    def construct_multiplicative_identity_element(self):
        if self.field_characteristic.bit_length() > 63:
            identity_coefficients = [0] * self.algebraic_degree
            identity_coefficients[0] = 1
        else:
            identity_coefficients = np.zeros(self.algebraic_degree, dtype=np.int64)
            identity_coefficients[0] = 1
        return self.instantiate_algebraic_element(identity_coefficients)

class AlgebraicElementRepresentation:
    def __init__(self, ambient_algebraic_structure, coefficient_vector):
        self.ambient_algebraic_structure = ambient_algebraic_structure
        self.coefficient_representation = self._normalize_coefficient_representation(coefficient_vector)
    
    def _normalize_coefficient_representation(self, coefficient_vector):
        field_characteristic = self.ambient_algebraic_structure.field_characteristic
        algebraic_degree = self.ambient_algebraic_structure.algebraic_degree
        
        if field_characteristic.bit_length() > 63:
            if isinstance(coefficient_vector, np.ndarray):
                coefficient_list = coefficient_vector.tolist()
            else:
                coefficient_list = list(coefficient_vector)
            
            modular_reduced_coefficients = [int(c) % field_characteristic for c in coefficient_list]
            normalized_array = np.array(modular_reduced_coefficients, dtype=object)
        else:
            normalized_array = np.array(coefficient_vector, dtype=np.int64) % field_characteristic
        
        if len(normalized_array) > algebraic_degree:
            normalized_array = normalized_array[:algebraic_degree]
        elif len(normalized_array) < algebraic_degree:
            if field_characteristic.bit_length() > 63:
                extended_array = np.zeros(algebraic_degree, dtype=object)
            else:
                extended_array = np.zeros(algebraic_degree, dtype=np.int64)
            extended_array[:len(normalized_array)] = normalized_array
            normalized_array = extended_array
        
        return normalized_array
    
    def decompose_via_chinese_remainder_theorem(self):
        decomposition_components = []
        for factorization_component in self.ambient_algebraic_structure.irreducible_factorization_components:
            component_evaluation = self._evaluate_modulo_irreducible_factor(factorization_component)
            decomposition_components.append(component_evaluation)
        return decomposition_components
    
    def _evaluate_modulo_irreducible_factor(self, irreducible_factor):
        if len(irreducible_factor) == 2 and irreducible_factor[0] == 1:
            evaluation_point = (-irreducible_factor[1]) % self.ambient_algebraic_structure.field_characteristic
            polynomial_evaluation = 0
            for degree_index, coefficient_value in enumerate(self.coefficient_representation):
                power_contribution = pow(evaluation_point, degree_index, self.ambient_algebraic_structure.field_characteristic)
                polynomial_evaluation = (polynomial_evaluation + coefficient_value * power_contribution) % self.ambient_algebraic_structure.field_characteristic
            return polynomial_evaluation
        else:
            return self.coefficient_representation.copy()
    
    @classmethod
    def reconstruct_from_chinese_remainder_decomposition(cls, ambient_structure, decomposition_components):
        coefficient_reconstruction = np.zeros(ambient_structure.algebraic_degree, dtype=np.int64)
        
        if len(decomposition_components) == 1:
            if isinstance(decomposition_components[0], (int, np.integer)):
                coefficient_reconstruction[0] = decomposition_components[0]
            else:
                coefficient_reconstruction[:len(decomposition_components[0])] = decomposition_components[0]
        else:
            for component_index, component_value in enumerate(decomposition_components):
                if isinstance(component_value, (int, np.integer)):
                    coefficient_reconstruction[component_index % ambient_structure.algebraic_degree] = (
                        coefficient_reconstruction[component_index % ambient_structure.algebraic_degree] + component_value
                    ) % ambient_structure.field_characteristic
                else:
                    for coefficient_index, coefficient_value in enumerate(component_value):
                        coefficient_reconstruction[coefficient_index % ambient_structure.algebraic_degree] = (
                            coefficient_reconstruction[coefficient_index % ambient_structure.algebraic_degree] + coefficient_value
                        ) % ambient_structure.field_characteristic
        
        return cls(ambient_structure, coefficient_reconstruction)
    
    def __add__(self, algebraic_operand):
        if isinstance(algebraic_operand, AlgebraicElementRepresentation):
            if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
                addition_result = []
                for left_coeff, right_coeff in zip(self.coefficient_representation, algebraic_operand.coefficient_representation):
                    sum_coefficient = (int(left_coeff) + int(right_coeff)) % self.ambient_algebraic_structure.field_characteristic
                    addition_result.append(sum_coefficient)
                result_coefficients = np.array(addition_result, dtype=object)
            else:
                result_coefficients = (self.coefficient_representation + algebraic_operand.coefficient_representation) % self.ambient_algebraic_structure.field_characteristic
        else:
            result_coefficients = self.coefficient_representation.copy()
            if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
                result_coefficients[0] = (int(result_coefficients[0]) + int(algebraic_operand)) % self.ambient_algebraic_structure.field_characteristic
            else:
                result_coefficients[0] = (result_coefficients[0] + algebraic_operand) % self.ambient_algebraic_structure.field_characteristic
        
        return AlgebraicElementRepresentation(self.ambient_algebraic_structure, result_coefficients)
    
    def __sub__(self, algebraic_operand):
        if isinstance(algebraic_operand, AlgebraicElementRepresentation):
            if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
                subtraction_result = []
                for left_coeff, right_coeff in zip(self.coefficient_representation, algebraic_operand.coefficient_representation):
                    difference_coefficient = (int(left_coeff) - int(right_coeff)) % self.ambient_algebraic_structure.field_characteristic
                    subtraction_result.append(difference_coefficient)
                result_coefficients = np.array(subtraction_result, dtype=object)
            else:
                result_coefficients = (self.coefficient_representation - algebraic_operand.coefficient_representation) % self.ambient_algebraic_structure.field_characteristic
        else:
            result_coefficients = self.coefficient_representation.copy()
            if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
                result_coefficients[0] = (int(result_coefficients[0]) - int(algebraic_operand)) % self.ambient_algebraic_structure.field_characteristic
            else:
                result_coefficients[0] = (result_coefficients[0] - algebraic_operand) % self.ambient_algebraic_structure.field_characteristic
        
        return AlgebraicElementRepresentation(self.ambient_algebraic_structure, result_coefficients)
    
    def __mul__(self, algebraic_operand):
        if isinstance(algebraic_operand, AlgebraicElementRepresentation):
            return self._execute_polynomial_multiplication(algebraic_operand)
        else:
            return self._execute_scalar_multiplication(algebraic_operand)
    
    def _execute_polynomial_multiplication(self, polynomial_operand):
        if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
            left_coefficients = [int(c) for c in self.coefficient_representation]
            right_coefficients = [int(c) for c in polynomial_operand.coefficient_representation]
            
            convolution_result_size = len(left_coefficients) + len(right_coefficients) - 1
            convolution_product = [0] * convolution_result_size
            for left_index, left_value in enumerate(left_coefficients):
                for right_index, right_value in enumerate(right_coefficients):
                    convolution_product[left_index + right_index] = (
                        convolution_product[left_index + right_index] + left_value * right_value
                    ) % self.ambient_algebraic_structure.field_characteristic
            
            reduced_result = [0] * self.ambient_algebraic_structure.algebraic_degree
        else:
            convolution_product = np.convolve(self.coefficient_representation, polynomial_operand.coefficient_representation)
            reduced_result = np.zeros(self.ambient_algebraic_structure.algebraic_degree, dtype=np.int64)
        
        for coefficient_index, coefficient_value in enumerate(convolution_product):
            if coefficient_index < self.ambient_algebraic_structure.algebraic_degree:
                if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
                    reduced_result[coefficient_index] = (reduced_result[coefficient_index] + int(coefficient_value)) % self.ambient_algebraic_structure.field_characteristic
                else:
                    reduced_result[coefficient_index] = (reduced_result[coefficient_index] + coefficient_value) % self.ambient_algebraic_structure.field_characteristic
            else:
                reduction_index = coefficient_index % self.ambient_algebraic_structure.algebraic_degree
                if self._is_cyclotomic_polynomial():
                    if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
                        reduced_result[reduction_index] = (reduced_result[reduction_index] - int(coefficient_value)) % self.ambient_algebraic_structure.field_characteristic
                    else:
                        reduced_result[reduction_index] = (reduced_result[reduction_index] - coefficient_value) % self.ambient_algebraic_structure.field_characteristic
                else:
                    if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
                        reduced_result[reduction_index] = (reduced_result[reduction_index] + int(coefficient_value)) % self.ambient_algebraic_structure.field_characteristic
                    else:
                        reduced_result[reduction_index] = (reduced_result[reduction_index] + coefficient_value) % self.ambient_algebraic_structure.field_characteristic
        
        if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
            reduced_result = np.array(reduced_result, dtype=object)
        
        return AlgebraicElementRepresentation(self.ambient_algebraic_structure, reduced_result)
    
    def _execute_scalar_multiplication(self, scalar_operand):
        if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
            scalar_product_result = []
            for coefficient_value in self.coefficient_representation:
                product_coefficient = (int(coefficient_value) * int(scalar_operand)) % self.ambient_algebraic_structure.field_characteristic
                scalar_product_result.append(product_coefficient)
            result_coefficients = np.array(scalar_product_result, dtype=object)
        else:
            result_coefficients = (self.coefficient_representation * scalar_operand) % self.ambient_algebraic_structure.field_characteristic
        
        return AlgebraicElementRepresentation(self.ambient_algebraic_structure, result_coefficients)
    
    def _is_cyclotomic_polynomial(self):
        polynomial_vector = self.ambient_algebraic_structure.polynomial_representation_vector
        return (len(polynomial_vector) == self.ambient_algebraic_structure.algebraic_degree + 1 and 
                polynomial_vector[-1] == 1 and 
                polynomial_vector[0] == 1 and 
                all(polynomial_vector[1:-1] == 0))
    
    def __rmul__(self, scalar_operand):
        return self.__mul__(scalar_operand)
    
    def __neg__(self):
        if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
            negation_result = []
            for coefficient_value in self.coefficient_representation:
                negated_coefficient = (-int(coefficient_value)) % self.ambient_algebraic_structure.field_characteristic
                negation_result.append(negated_coefficient)
            result_coefficients = np.array(negation_result, dtype=object)
        else:
            result_coefficients = (-self.coefficient_representation) % self.ambient_algebraic_structure.field_characteristic
        
        return AlgebraicElementRepresentation(self.ambient_algebraic_structure, result_coefficients)
    
    def generate_discrete_gaussian_error_instance(self, standard_deviation_parameter=1.0):
        gaussian_random_values = np.random.normal(0, standard_deviation_parameter, self.ambient_algebraic_structure.algebraic_degree)
        
        if self.ambient_algebraic_structure.field_characteristic.bit_length() > 63:
            discretized_coefficients = []
            for random_value in gaussian_random_values:
                discretized_value = int(np.rint(random_value))
                discretized_coefficients.append(discretized_value)
            return AlgebraicElementRepresentation(self.ambient_algebraic_structure, discretized_coefficients)
        else:
            discretized_coefficients = np.rint(gaussian_random_values).astype(np.int64)
            return AlgebraicElementRepresentation(self.ambient_algebraic_structure, discretized_coefficients)
    
    def __repr__(self):
        return f"AlgebraicElementRepresentation({self.coefficient_representation})"
    
    def __str__(self):
        return str(self.coefficient_representation)

class CyclotomicPolynomialRingConstructor:
    @staticmethod
    def construct_cyclotomic_ring_structure(algebraic_degree, field_characteristic):
        if field_characteristic.bit_length() > 63:
            polynomial_coefficients = [0] * (algebraic_degree + 1)
            polynomial_coefficients[0] = 1
            polynomial_coefficients[-1] = 1
        else:
            polynomial_coefficients = np.zeros(algebraic_degree + 1, dtype=np.int64)
            polynomial_coefficients[0] = 1
            polynomial_coefficients[-1] = 1
        
        return AbstractAlgebraicStructureRepresentation(field_characteristic, polynomial_coefficients)

class BinaryDecompositionMatrixGenerator:
    @staticmethod
    def construct_binary_decomposition_identity_matrix(matrix_dimension):
        logarithmic_dimension = ceil(log2(matrix_dimension)) if matrix_dimension > 1 else 1
        identity_matrix = np.zeros((matrix_dimension, matrix_dimension), dtype=np.int64)
        
        for row_index in range(matrix_dimension):
            for column_index in range(min(logarithmic_dimension, matrix_dimension)):
                if row_index == column_index:
                    identity_matrix[row_index][column_index] = 1
                elif row_index == column_index + logarithmic_dimension:
                    identity_matrix[row_index][column_index] = 1
        
        for diagonal_index in range(matrix_dimension):
            for off_diagonal_index in range(matrix_dimension):
                if identity_matrix[diagonal_index][off_diagonal_index] == 0 and diagonal_index == off_diagonal_index:
                    identity_matrix[diagonal_index][off_diagonal_index] = 1
        
        return identity_matrix

class TextualRepresentationParser:
    @staticmethod
    def parse_textual_array_representation(textual_representation):
        parsed_array = []
        for row_representation in textual_representation.split('\n'):
            if row_representation.strip() != '':
                sanitized_row = row_representation.replace('[', '').replace(']', '').strip()
                parsed_array.append([int(x) for x in sanitized_row.split()])
        return np.array(parsed_array, dtype=np.int64)

class ObjectMemoryFootprintEstimator:
    @staticmethod
    def estimate_object_memory_consumption(target_object):
        import sys
        if hasattr(target_object, 'coefficient_representation'):
            return target_object.coefficient_representation.nbytes + sys.getsizeof(target_object)
        elif isinstance(target_object, np.ndarray):
            return target_object.nbytes
        elif isinstance(target_object, list):
            total_size = sys.getsizeof(target_object)
            for element in target_object:
                total_size += ObjectMemoryFootprintEstimator.estimate_object_memory_consumption(element)
            return total_size
        elif isinstance(target_object, (int, float)):
            return 8
        else:
            return sys.getsizeof(target_object)

performance_metrics_aggregator = TemporalPerformanceMetricsAggregator()
cryptographic_prime_oracle = CryptographicPrimeGenerationOracle()

def stat(operational_descriptor):
    performance_metrics_aggregator.register_computational_checkpoint(operational_descriptor)

def powmod(base_operand, exponent_parameter, modular_parameter):
    modular_arithmetic_engine = ModularArithmeticComputationalEngine()
    return modular_arithmetic_engine.compute_modular_exponentiation_via_binary_method(
        base_operand, exponent_parameter, modular_parameter
    )

def is_prime(candidate_value):
    primality_tester = ProbabilisticPrimalityTestingFramework()
    return primality_tester.execute_miller_rabin_compositeness_verification(candidate_value)

def gen_prime(bit_length_specification):
    return cryptographic_prime_oracle.synthesize_prime_with_specified_bit_length(bit_length_specification)

def generateSophieGermainPrime(bit_length_specification):
    return cryptographic_prime_oracle.generate_sophie_germain_prime_configuration(bit_length_specification)

def create_cyclotomic_ring(algebraic_degree, field_characteristic):
    return CyclotomicPolynomialRingConstructor.construct_cyclotomic_ring_structure(
        algebraic_degree, field_characteristic
    )

def binary_decompose_identity(matrix_dimension):
    return BinaryDecompositionMatrixGenerator.construct_binary_decomposition_identity_matrix(matrix_dimension)

def text2array(textual_representation):
    return TextualRepresentationParser.parse_textual_array_representation(textual_representation)

def get_object_size_bytes(target_object):
    return ObjectMemoryFootprintEstimator.estimate_object_memory_consumption(target_object)

RingElement = AlgebraicElementRepresentation
PolynomialRing = AbstractAlgebraicStructureRepresentation
