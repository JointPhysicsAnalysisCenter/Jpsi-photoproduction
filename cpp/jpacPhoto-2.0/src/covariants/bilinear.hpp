// Here we assemble dirac_spinors together into Lorentz covariant bilinears
// e.g. ubar gamma_mu u
//
// This involves defining a new special-case of the lorentz_tensor class
// which arises when an existing lorentz_tensor<dirac_matrix> is combined with a spinor
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef BILINEAR_HPP
#define BILINEAR_HPP

#include "constants.hpp"
#include "contract.hpp"
#include "lorentz_tensor.hpp"
#include "dirac_matrix.hpp"
#include "dirac_spinor.hpp"

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // Covariant gamma matrix structures

    // Non-trivial dirac_matrix tensors
    // Any arbitrary tensor up to rank-2 can be built from these

    inline lorentz_tensor<dirac_matrix,1> gamma_vector()
    { 
        return lorentz_vector<dirac_matrix>({gamma_0(), gamma_1(), gamma_2(), gamma_3()});
    };
    inline lorentz_tensor<dirac_matrix,2> sigma_tensor()
    { 
        return tensor_product(gamma_vector(), gamma_vector()) - identity<dirac_matrix>()*metric_tensor(); 
    };

    inline dirac_matrix slash(lorentz_tensor<complex,1> q)
    {
        auto gamma = gamma_vector();
        dirac_matrix sum = zero<dirac_matrix>();
        for (auto mu : LORENTZ_INDICES)
        {
            sum += metric(mu) * gamma(mu) * q(mu);
        }
        return sum;
    };

    // ---------------------------------------------------------------------------
    // Special case of lorentz_tensor which mixes spinors and matrices. 
    // Here the saved subtensors, normalizations, and return values are different types
    
    template<int Rank>
    class bilinear_tensor : public tensor_object<complex>
    {
        public: 

        // Default tensor with nothing initialized
        bilinear_tensor<Rank>(){};

        // Copy constructor
        bilinear_tensor<Rank>(bilinear_tensor<Rank> const & old)
        : _matrix(old._matrix),
          _lhs(old._lhs), _rhs(old._rhs)
        {};

        // Implicit constructor, stores pointers to constituent tensors of smaller rank
        bilinear_tensor<Rank>(dirac_spinor L, lorentz_tensor<dirac_matrix, Rank> Ts, dirac_spinor R)
        :   _lhs(L), _rhs(R),
            _matrix(Ts)
        {};

        inline complex operator()(std::vector<lorentz_index> indices)
        {
            if (indices.size() != Rank) return error("lorentz_tensor", "Incorrect number of indices passed!", NaN<complex>());          

            // begin producting all the matrices to get one 
            dirac_matrix M = _matrix(indices);
            return contract(_lhs, M*_rhs); 
        };

        // Get the rank of the tensor (number of open indicies)
        const inline int rank(){ return Rank; };

        protected:

        // These are always treated as if they were tensor products 
        lorentz_tensor<dirac_matrix, Rank> _matrix;

        // Normalizations for the spinors on either side
        dirac_spinor _lhs = zero<dirac_spinor>();
        dirac_spinor _rhs = zero<dirac_spinor>();
    };

    // ---------------------------------------------------------------------------
    // Bilinear methods combine two dirac_spiors with a lorentz_tensor 

    template<int R>
    inline lorentz_tensor<complex,R> bilinear(dirac_spinor ubar, lorentz_tensor<dirac_matrix,R> Gamma, dirac_spinor u)
    {
        std::shared_ptr<tensor_object<complex>> m_ptr = std::make_shared<bilinear_tensor<R>>(ubar, Gamma, u);
        
        return lorentz_tensor<complex,R>({m_ptr}, false);
    };
};

#endif