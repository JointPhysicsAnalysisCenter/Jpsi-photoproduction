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

#ifndef DIRAC_SPINOR_HPP
#define DIRAC_SPINOR_HPP

#include <vector> 
#include <array>

#include "constants.hpp"
#include "dirac_matrix.hpp"

namespace jpacPhoto
{
    // -----------------------------------------------------------------------
    // The actual spinor object behave basically like 4-component vectors

    class dirac_spinor
    {
        public:

        // Initialize empty
        dirac_spinor(){};

        // Initialize with given constant components
        dirac_spinor(std::array<complex,4> entries)
        : _entries(entries)
        {};
        
        // Access an element
        complex operator()(dirac_index a);
        inline complex operator()(int a){return operator()(static_cast<dirac_index>(a));};

        // Reassignment
        dirac_spinor & operator=(dirac_spinor const & u);
        dirac_spinor & operator*=(complex c);
        dirac_spinor & operator/=(complex c);
        dirac_spinor & operator+=(dirac_spinor u);
        dirac_spinor & operator*=(dirac_spinor u);

        // Negate
        dirac_spinor operator-();

        // Take Dirac Adjoint
        dirac_spinor adjoint();

        private: 

        // Stored data
        std::array<complex,4> _entries;
    };

    // Arbitrary dirac-space operators can be built from the above as linear combinations
    dirac_spinor operator+(dirac_spinor lhs, dirac_spinor rhs);
    dirac_spinor operator-(dirac_spinor lhs, dirac_spinor rhs);

    // Multiply by a constant
    dirac_spinor operator*(complex c, dirac_spinor rhs);
    dirac_spinor operator*(dirac_spinor rhs, complex c);
    dirac_spinor operator/(dirac_spinor rhs, complex c);

    // Get adjoint from outside
    inline dirac_spinor conj(dirac_spinor x){ return x.adjoint(); };

    // Spinor filled with NaN's for error throwing
    template<> 
    inline dirac_spinor NaN<dirac_spinor>()
    { return NaN<complex>() * dirac_spinor({1, 1, 1, 1}); }

    template<>
    inline dirac_spinor identity() { return dirac_spinor({1, 1, 1, 1}); };

    template<>
    inline dirac_spinor zero() { return dirac_spinor({0, 0, 0, 0}); };

    // Multiplication of spinor objects element-wise
    // This is kind of weird, but required to make lorentz_tensors<dirac_spinors> work 
    // as they should. 
    // This is NOT contracting the spinors (ubar . u) but instead multipling them element
    // wise: (u*v) = {u0*v0, u1*v1, u2*v2, u3*v3}
    dirac_spinor operator*(dirac_spinor u, dirac_spinor v);
    
    // ---------------------------------------------------------------------------
    // Interactions with matrices
    
    // Multiply on the right by a dirac_matrix
    dirac_spinor operator*(dirac_spinor ubar, dirac_matrix M);

    // Multiply on the left by a dirac_matrix
    dirac_spinor operator*(dirac_matrix M, dirac_spinor u);
};

#endif