// The dirac_spinor class more broadly defines a 4-component, vector-like object
//  which is assumed to have the appropriate transformations in spinor space.
//
// The usual spin-1/2 wave functions e.g. are specific instances of this more 
// general object. 
// 
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#include "dirac_spinor.hpp"
#include "dirac_matrix.hpp"

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // Intrinsic properties of dirac_spinors

    // reassignment 
    dirac_spinor & dirac_spinor::operator=(dirac_spinor const & u)
    {
        _entries = u._entries;
        return *this;
    };

    // Access elements
    complex dirac_spinor::operator()(dirac_index a)
    {
        return _entries[+a];
    };

    // Scaled re-assignement
    dirac_spinor & dirac_spinor::operator*=(complex c)
    {
        for (auto i : DIRAC_INDICES)
        {
            _entries[+i] *= c;
        }
        return *this;
    }

    dirac_spinor & dirac_spinor::operator/=(complex c)
    {
        for (auto i : DIRAC_INDICES)
        {
            _entries[+i] /= c;
        }
        return *this;
    };

    dirac_spinor & dirac_spinor::operator+=(dirac_spinor u)
    {
        for (auto i : DIRAC_INDICES)
        {
            _entries[+i] += u(i);
        };
        return *this;
    };

    dirac_spinor & dirac_spinor::operator*=(dirac_spinor u)
    {
        for (auto i : DIRAC_INDICES)
        {
            _entries[+i] *= u(i);
        };
        return *this;
    };

    // Negate a spinor
    dirac_spinor dirac_spinor::operator-()
    {
        dirac_spinor neg = *this;
        for (auto i : DIRAC_INDICES)
        {
            neg._entries[+i] *= -1;
        }
        return neg;
    };

    // Return the adjoint of a spinor
    // Because we dont actually keep track of the orientation of the vector
    // i.e. row vs column vector
    // Its assumed the used will keep track of this in the elements
    // This simply takes the conjugate and multiplies by gamma0
    dirac_spinor dirac_spinor::adjoint()
    {
        dirac_spinor conj = *this;
        for (auto x : conj._entries)
        {
            x = std::conj(x);
        };
        
        return conj * gamma_0();
    };

    // ---------------------------------------------------------------------------
    // Non-member operations of dirac_spinors

    //Add two spinors together
    dirac_spinor operator+(dirac_spinor lhs, dirac_spinor rhs)
    {
        std::array<complex,4> entries;
        for (auto i : DIRAC_INDICES)
        {
            entries[+i] = lhs(i) + rhs(i);
        };
        return dirac_spinor(entries);
    };

    // Subtract two spinors 
    dirac_spinor operator-(dirac_spinor lhs, dirac_spinor rhs)
    {
        return lhs + (-rhs);
    };

    // Multiply by a constant
    dirac_spinor operator*(complex c, dirac_spinor rhs)
    {
        dirac_spinor copy = rhs;
        copy *= c;
        return copy;
    };

    dirac_spinor operator*(dirac_spinor lhs, complex c)
    {
        return c * lhs;
    };

    dirac_spinor operator/(dirac_spinor lhs, complex c)
    {
        return (1./c) * lhs;
    };
    
    // Element-wise multiplication
    dirac_spinor operator*(dirac_spinor lhs, dirac_spinor rhs)
    {
        std::array<complex,4> entries;
        for (auto i : DIRAC_INDICES)
        {
            entries[+i] = lhs(i) * rhs(i);
        };
        return dirac_spinor(entries);
    };
    
    // ---------------------------------------------------------------------------
    // Interactions with spinors
    
     // Multiply on the right by a dirac_matrix
    dirac_spinor operator*(dirac_spinor ubar, dirac_matrix M)
    {
        std::array<complex,4> entries;
        for (auto i : DIRAC_INDICES)
        {
            complex sum = 0.;
            for (auto j : DIRAC_INDICES)
            {
                sum += ubar(j) * M(j,i);
            };
            entries[+i] = sum;
        };

        return dirac_spinor(entries);
    };

    // Multiply on the left by a dirac_matrix
    dirac_spinor operator*(dirac_matrix M, dirac_spinor u)
    {
        std::array<complex,4> entries;
        for (auto i : DIRAC_INDICES)
        {
            complex sum = 0.;
            for (auto j : DIRAC_INDICES)
            {
                sum += M(i,j) * u(j);
            };
            entries[+i] = sum;
        };

        return dirac_spinor(entries);
    };
};