from flask import Flask, render_template, request
from sympy import Matrix, symbols, sympify, Poly
import numpy as np
from sympy.parsing.latex import parse_latex

app = Flask(__name__)

def matrix_to_latex(matrix):
    if matrix is None:
        return None
    latex = r'\begin{bmatrix}'
    for row in matrix:
        latex += ' & '.join(map(str, row)) + r' \\ '
    latex = latex.rstrip(r' \\ ')  # Remove the last row separator
    latex += r'\end{bmatrix}'
    return latex

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', matrix=None, basis=None, error=None)

@app.route('/calculate', methods=['POST'])
def linear_map_to_matrix():
    if request.method == 'POST':
        latex_input_lm = request.form['transformation']
        basis_input = request.form['basis']
        error_message = None
        np_matrix = None
        basis_vectors = None

        try:
            # Parse LaTeX input into a SymPy Matrix for the linear transformation
            sympy_expr = parse_latex(latex_input_lm)
            sympy_linear_map = Poly(sympy_expr)
            x = symbols('x')  # Define the symbol 'x' for polynomial basis

            # Parse basis input
            basis_list = [sympify(expr, locals={'x': x}, convert_xor=True) for expr in basis_input.split(',')]
            basis_vectors = basis_list
            dim = len(basis_vectors)
            print(dim)

            # Apply transformation and express in original basis
            matrix_columns = []
            for vec in basis_vectors:
                transformed_vec = sympy_linear_map.subs(x, vec)
                standard_basis_vec = Matrix([[coeff] for coeff in Poly(transformed_vec, x).all_coeffs()])
                standard_basis_vec = standard_basis_vec.col_join(Matrix.zeros(dim - standard_basis_vec.rows, 1))  # Zero padding for missing terms
                matrix_columns.append(standard_basis_vec)

            # Constructing the matrix representation
            matrix_representation = Matrix.hstack(*matrix_columns)
            np_matrix = np.array(matrix_representation).astype(np.float64)

            if np_matrix is not None:
                print("Matrix computed:", np_matrix)
            else:
                print("Matrix computation resulted in None")
            eigenvalues, eigenvectors = np.linalg.eig(np_matrix)

            # Convert eigenvalues and eigenvectors to lists (for template compatibility)
            eigenvalues_list = eigenvalues.tolist()
            eigenvectors_list = [vec.tolist() for vec in eigenvectors.T]

            # Set flags for the existence of eigenvalues and eigenvectors
            has_eigenvalues = bool(eigenvalues_list)
            has_eigenvectors = bool(eigenvectors_list)

        except Exception as e:
            error_message = f'Error in processing: {e}'
            eigenvalues_list = []
            eigenvectors_list = []
            has_eigenvalues = False
            has_eigenvectors = False

        return render_template('index.html', latex_matrix= matrix_to_latex(np_matrix), basis=basis_vectors, eigenvalues=eigenvalues_list, eigenvectors=eigenvectors_list, has_eigenvalues=has_eigenvalues, has_eigenvectors=has_eigenvectors, error=error_message)

    return render_template('index.html', latex_matrix= None, basis=None, eigenvalues=[], eigenvectors=[], has_eigenvalues=False, has_eigenvectors=False, error=None)
if __name__ == '__main__':
    app.run(debug=True)