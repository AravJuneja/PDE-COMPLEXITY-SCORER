import sympy as sp
from sympy import symbols, sympify, diff, latex
from sympy.parsing.sympy_parser import parse_expr
import re
from typing import Dict, List, Optional, Tuple
import numpy as np


class PDEComplexityScorer:
    
    def __init__(self, equation: str, domain: str, boundary_conditions: str, initial_conditions: Optional[str] = None):
        self.equation_str = equation
        self.domain = domain
        self.bc = boundary_conditions
        self.ic = initial_conditions
        
        self.variables = self._extract_variables()
        self.pde = self._parse_equation()
        
        self.scores = {}
        self.explanations = {}
        
    def _extract_variables(self) -> List[str]:
        potential_vars = ['t', 'x', 'y', 'z', 'r', 'theta', 'phi']
        found_vars = []
        
        equation_lower = self.equation_str.lower()
        domain_lower = self.domain.lower()
        
        for var in potential_vars:
            if var in equation_lower or var in domain_lower:
                found_vars.append(var)
                
        return found_vars
    
    def _parse_equation(self):
        eq_str = self.equation_str
        
        eq_str = eq_str.replace('laplacian(u)', 'u_xx + u_yy')
        eq_str = eq_str.replace('∇²u', 'u_xx + u_yy')
        eq_str = eq_str.replace('Δu', 'u_xx + u_yy')
        eq_str = eq_str.replace('nabla', 'u_xx + u_yy')
        
        eq_str = eq_str.replace('u_tt', 'diff(u, t, t)')
        eq_str = eq_str.replace('u_xx', 'diff(u, x, x)')
        eq_str = eq_str.replace('u_yy', 'diff(u, y, y)')
        eq_str = eq_str.replace('u_t', 'diff(u, t)')
        eq_str = eq_str.replace('u_x', 'diff(u, x)')
        eq_str = eq_str.replace('u_y', 'diff(u, y)')
        
        eq_str = eq_str.replace('^', '**')
        
        try:
            parsed = sympify(eq_str)
            return parsed
        except:
            return eq_str
    
    def score_dimensionality(self) -> int:
        spatial_dims = 0
        temporal_dims = 0
        
        if 't' in self.variables:
            temporal_dims = 1
            
        for var in ['x', 'y', 'z']:
            if var in self.variables:
                spatial_dims += 1
        
        for var in ['r', 'theta', 'phi']:
            if var in self.variables:
                spatial_dims = max(spatial_dims, 2)
                break
        
        total = spatial_dims + temporal_dims
        
        if total == 1:
            score = 1
        elif total == 2:
            score = 2
        elif total == 3:
            score = 3
        else:
            score = 4
            
        self.explanations['dimensionality'] = f"Spatial dims: {spatial_dims}, Temporal dims: {temporal_dims}"
        return score
    
    def score_nonlinearity(self) -> int:
        eq_str = self.equation_str.lower()
        
        if 'exp(u)' in eq_str or 'sin(u)' in eq_str or 'cos(u)' in eq_str or 'log(u)' in eq_str:
            self.explanations['nonlinearity'] = "Detected transcendental function"
            return 4
        
        polynomial_pattern = r'u\*\*(\d+)|u\^(\d+)'
        matches = re.findall(polynomial_pattern, eq_str)
        
        max_degree = 0
        for match in matches:
            degree = int(match[0] if match[0] else match[1])
            max_degree = max(max_degree, degree)
        
        if max_degree >= 3:
            self.explanations['nonlinearity'] = f"Polynomial term: u^{max_degree}"
            return 3
        
        if 'u*u_x' in eq_str or 'u*u_y' in eq_str or 'u·∇u' in eq_str or 'u*∇u' in eq_str or 'u*nabla' in eq_str:
            self.explanations['nonlinearity'] = "Bilinear advection term"
            return 2
            
        if max_degree == 2:
            self.explanations['nonlinearity'] = "Quadratic term: u^2"
            return 2
        
        self.explanations['nonlinearity'] = "Linear equation"
        return 0
    
    def score_boundary_complexity(self) -> int:
        bc_lower = self.bc.lower()
        
        if 'irregular' in bc_lower or 'complex geometry' in bc_lower:
            self.explanations['boundary'] = "Irregular geometry"
            return 4
        
        if 'moving' in bc_lower or 'time-dependent' in bc_lower or 'lid' in bc_lower:
            self.explanations['boundary'] = "Time-dependent boundary"
            return 3
        
        dirichlet = 'dirichlet' in bc_lower or 'u=' in bc_lower or 'u(' in bc_lower
        neumann = 'neumann' in bc_lower or 'u_n' in bc_lower or 'du/dn' in bc_lower
        robin = 'robin' in bc_lower
        mixed = 'mixed' in bc_lower
        
        if (dirichlet and neumann) or robin or mixed:
            self.explanations['boundary'] = "Mixed boundary conditions"
            return 2
        
        if dirichlet:
            self.explanations['boundary'] = "Simple Dirichlet"
            return 1
        
        self.explanations['boundary'] = "Simple boundary conditions"
        return 1
    
    def score_time_dependence(self) -> int:
        if 't' not in self.variables:
            self.explanations['time'] = "Steady-state (no time dependence)"
            return 0
        
        eq_str = self.equation_str.lower()
        
        if 'u_tt' in eq_str or 'd²u/dt²' in eq_str or 'diff(u, t, t)' in eq_str:
            self.explanations['time'] = "Second-order time derivative"
            return 2
        
        if 'u_ttt' in eq_str:
            self.explanations['time'] = "Higher-order time derivative (stiff)"
            return 3
        
        if 'u_t' in eq_str or 'du/dt' in eq_str or 'diff(u, t)' in eq_str:
            self.explanations['time'] = "First-order time derivative"
            return 1
        
        self.explanations['time'] = "Time-dependent domain"
        return 1
    
    def score_coupling(self) -> int:
        eq_str = self.equation_str.lower()
        bc_str = self.bc.lower()
        
        dependent_vars = []
        for var in ['u', 'v', 'w', 'p', 'phi', 'psi']:
            if var in eq_str or var in bc_str:
                dependent_vars.append(var)
        
        dependent_vars = list(set(dependent_vars))
        
        num_vars = len(dependent_vars)
        
        if num_vars == 1:
            self.explanations['coupling'] = "Single equation"
            return 0
        elif num_vars == 2:
            self.explanations['coupling'] = "Weak coupling (2 variables)"
            return 1
        elif num_vars <= 4:
            self.explanations['coupling'] = f"Strong coupling ({num_vars} variables)"
            return 2
        else:
            self.explanations['coupling'] = f"Complex coupling ({num_vars}+ variables)"
            return 3
    
    def get_total_score(self) -> Dict:
        self.scores['dimensionality'] = self.score_dimensionality()
        self.scores['nonlinearity'] = self.score_nonlinearity()
        self.scores['boundary'] = self.score_boundary_complexity()
        self.scores['time'] = self.score_time_dependence()
        self.scores['coupling'] = self.score_coupling()
        self.scores['total'] = sum([
            self.scores['dimensionality'],
            self.scores['nonlinearity'],
            self.scores['boundary'],
            self.scores['time'],
            self.scores['coupling']
        ])
        
        return self.scores
    
    def explain_scores(self) -> str:
        if not self.scores:
            self.get_total_score()
        
        explanation = f"\n{'='*60}\n"
        explanation += f"PDE COMPLEXITY ANALYSIS\n"
        explanation += f"{'='*60}\n\n"
        explanation += f"Equation: {self.equation_str}\n"
        explanation += f"Domain: {self.domain}\n\n"
        
        for key, value in self.scores.items():
            if key != 'total':
                explanation += f"{key.capitalize()}: {value}\n"
                if key in self.explanations:
                    explanation += f"  → {self.explanations[key]}\n"
        
        explanation += f"\n{'='*60}\n"
        explanation += f"TOTAL COMPLEXITY SCORE: {self.scores['total']}\n"
        explanation += f"{'='*60}\n"
        
        return explanation
    
    @classmethod
    def from_text(cls, text: str):
        equation = ""
        domain = ""
        boundary = ""
        
        lines = text.split('\n')
        for line in lines:
            if '=' in line and any(c in line for c in ['u', 'v', 'p']):
                equation = line.strip()
            if 'domain' in line.lower() or '[' in line:
                domain = line.strip()
            if 'boundary' in line.lower() or 'bc' in line.lower():
                boundary = line.strip()
        
        if not equation:
            equation = text
        if not domain:
            domain = "Unknown"
        if not boundary:
            boundary = "Unknown"
            
        return cls(equation, domain, boundary)


def score_qcpinn_paper_pdes():
    
    helmholtz = PDEComplexityScorer(
        equation="laplacian(u) + k**2 * u = f",
        domain="[-1, 1] x [-1, 1]",
        boundary_conditions="u(x,y) = h(x,y) on boundary"
    )
    
    wave = PDEComplexityScorer(
        equation="u_tt - c**2 * u_xx = 0",
        domain="[0, 1] x [0, 1]",
        boundary_conditions="u(0,t) = 0, u(1,t) = 0"
    )
    
    klein_gordon = PDEComplexityScorer(
        equation="u_tt - alpha*u_xx + beta*u + gamma*u**3 = 0",
        domain="[0, 1] x [0, 1]",
        boundary_conditions="u(0,t) = 0, u(1,t) = cos(5*pi*t) + t**3"
    )
    
    convection_diffusion = PDEComplexityScorer(
        equation="u_t + c1*u_x + c2*u_y - D*(u_xx + u_yy) = 0",
        domain="[0, 1] x [0, 1] x [0, 1]",
        boundary_conditions="u(t,x,y) = g(t,x,y) on boundary"
    )
    
    cavity = PDEComplexityScorer(
        equation="u_t + u*u_x + v*u_y = -p_x + mu*(u_xx + u_yy), v_t + u*v_x + v*v_y = -p_y + mu*(v_xx + v_yy), u_x + v_y = 0",
        domain="[0, 1] x [0, 1] x [0, 10]",
        boundary_conditions="u=0 on walls, u=1 on moving lid, no-slip"
    )
    
    pdes = {
        'Helmholtz': helmholtz,
        'Wave': wave,
        'Klein-Gordon': klein_gordon,
        'Convection-Diffusion': convection_diffusion,
        'Lid-Driven Cavity': cavity
    }
    
    results = []
    for name, pde in pdes.items():
        scores = pde.get_total_score()
        scores['name'] = name
        results.append(scores)
        print(pde.explain_scores())
    
    return results


if __name__ == "__main__":
    results = score_qcpinn_paper_pdes()
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'PDE':<25} {'Dim':<5} {'Nonlin':<8} {'Bound':<7} {'Time':<6} {'Coup':<6} {'Total':<6}")
    print("-"*60)
    
    for result in results:
        print(f"{result['name']:<25} {result['dimensionality']:<5} {result['nonlinearity']:<8} {result['boundary']:<7} {result['time']:<6} {result['coupling']:<6} {result['total']:<6}")