// Dirac matrixes, are 4x4 (rank-2) objects in spinor space and act on spinors
// Effectively they are operators, e.g. gamma matrices 
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#include "dirac_matrix.hpp"

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // Intrinsic operators of dirac_matrices

    // Assignment and scaled re-assignment 

    dirac_matrix & dirac_matrix::operator=(dirac_matrix const & G)
    {
        _N       = G._N;
        _entries = G._entries;
        return *this;
    };

    dirac_matrix & dirac_matrix::operator*=(complex c)
    {
        _N *= c;
        return *this;  
    };

    dirac_matrix & dirac_matrix::operator/=(complex c)
    {
        _N /= c;
        return *this;  
    };

    dirac_matrix & dirac_matrix::operator+=(dirac_matrix G)
    {
        std::array<std::array<complex,4>,4> new_entries;
        for (auto i : DIRAC_INDICES)
        {
            for (auto j : DIRAC_INDICES)
            {
                new_entries[+i][+j] = _N*_entries[+i][+j] + G(i,j);
            };
        };
        _entries = new_entries; _N = 1;
        return *this;  
    };

    
    // Multiply two matrices together
    dirac_matrix & dirac_matrix::operator*=(dirac_matrix rhs)
    {
        std::array<std::array<complex,4>,4> old_entries = _entries;

        for (auto i : DIRAC_INDICES)
        {
            for (auto j : DIRAC_INDICES)
            {
                complex sum = 0.;
                for (auto k : DIRAC_INDICES)
                {
                    sum += old_entries[+i][+k]*rhs(k,j);
                }
                _entries[+i][+j] = sum;
            };
        };

        return *this;
    };

    // Negation
    dirac_matrix dirac_matrix::operator-()
    {
        dirac_matrix neg = *this;
        neg *= -1;
        return neg;
    }; 
    
    // Return the hermitian adjoint of operator;
    dirac_matrix dirac_matrix::adjoint()
    {
        std::array<std::array<complex,4>,4> entries;

        for (auto i : DIRAC_INDICES)
        {
            for (auto j : DIRAC_INDICES)
            {
                entries[+i][+j] = std::conj(_entries[+j][+i]);
            };
        };

        return dirac_matrix(entries);
    };

    // ---------------------------------------------------------------------------
    // "Constructor" functions for Dirac matrices

    dirac_matrix gamma_0()
    {
        return dirac_matrix({{ { 1,  0,   0,   0},
                               { 0,  1,   0,   0},
                               { 0,  0,  -1,   0},
                               { 0,  0,   0,  -1}  }});
    };

    dirac_matrix gamma_1()
    {
        return dirac_matrix({{ { 0,  0,   0,   1},
                               { 0,  0,   1,   0},
                               { 0, -1,   0,   0},
                               {-1,  0,   0,   0}  }});
    };

    dirac_matrix gamma_2()
    {
        return dirac_matrix({{ { 0,  0,   0,  -I},
                               { 0,  0,   I,   0},
                               { 0,  I,   0,   0},
                               {-I,  0,   0,   0}  }});
    };

    dirac_matrix gamma_3()
    {
        return dirac_matrix({{ { 0,  0,   1,   0},
                               { 0,  0,   0,  -1},
                               {-1,  0,   0,   0},
                               { 0,  1,   0,   0}  }});
    };

    dirac_matrix gamma_5()
    {
        return dirac_matrix({{ { 0,  0,   1,   0},
                               { 0,  0,   0,   1},
                               { 1,  0,   0,   0},
                               { 0,  1,   0,   0}  }});
    };

    // ---------------------------------------------------------------------------
    // Interactions between matrices
    // Linear combination operators to build arbitrarty dirac operator

    // Add two matrices together
    dirac_matrix operator+(dirac_matrix lhs, dirac_matrix rhs)
    {
        std::array<std::array<complex,4>,4> entries;

        for (auto i : DIRAC_INDICES)
        {
            for (auto j : DIRAC_INDICES)
            {
                entries[i][j] = lhs(i,j) + rhs(i,j);
            };
        };

        return dirac_matrix(entries);
    };

    // Subtract two matrices
    dirac_matrix operator-(dirac_matrix lhs, dirac_matrix rhs)
    {
        return lhs + (-rhs);
    };

    // Multiply be a constant
    dirac_matrix operator*(complex c, dirac_matrix T)
    {
        dirac_matrix cT = T;
        cT._N *= c;
        return cT;
    };

    // Multiply two matrices together
    dirac_matrix operator*(dirac_matrix lhs, dirac_matrix rhs)
    {
        std::array<std::array<complex,4>,4> entries;

        for (auto i : DIRAC_INDICES)
        {
            for (auto j : DIRAC_INDICES)
            {
                complex sum = 0.;
                for (auto k : DIRAC_INDICES)
                {
                    sum += lhs(i,k)*rhs(k,j);
                }
                entries[+i][+j] = sum;
            };
        };

        return dirac_matrix(entries);
    };
};