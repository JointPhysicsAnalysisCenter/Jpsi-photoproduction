// The contract() function assembles different tensors into other structures
// At present these different interacitons must all be specified individually
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef CONTRACT_HPP
#define CONTRACT_HPP

#include "lorentz_tensor.hpp"
#include "dirac_spinor.hpp"
#include "dirac_matrix.hpp"

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // Contraction of dirac_spinors is different because theyre dirac_indices
    
    complex contract(dirac_spinor left, dirac_spinor right);

    // ---------------------------------------------------------------------------
    // Function to return a vector of all permutations of N indices

    std::vector<std::vector<lorentz_index>> permutations(unsigned N);
    
    // Single function to produce the metric along the diagonal
    int metric(lorentz_index mu);

    // Or given a set of permutations gives the product of arbitrary number of individual metrics
    int metric(std::vector<lorentz_index> permutations);

    // ---------------------------------------------------------------------------
    // I wrote this and it wors fine but you need to specify the template parameter 
    // which is annoying and i dont like it
    // e.g. i want to write contract(q, p) not contract<complex>(q, p);

    // template<class LType, class RType, int R>
    // inline auto contract(lorentz_tensor<LType,R> left, lorentz_tensor<RType,R> right)
    // {
    //     auto sum = 0 * identity<LType>() * identity<RType>();
    //     for (auto perm : permutations(R))
    //     {
    //         sum += metric(perm) * contract(left(perm), right(perm));
    //     };
    //     return sum;
    // };

    // // Two scalar vectors contracted 
    // inline complex contract(lorentz_tensor<complex,1> left, lorentz_tensor<complex,1> right)
    // {
    //     complex sum = 0;
    //     for (auto mu : LORENTZ_INDICES)
    //     {
    //         sum += metric(mu) * contract(left(mu), right(mu));
    //     };
    //     return sum;
    // };

    // ---------------------------------------------------------------------------
    // SMALL CASES ARE CODED BY HAND >:)

    // Two vectors contracted 
    template<class Type>
    inline Type contract(lorentz_tensor<Type,1> left, lorentz_tensor<Type,1> right)
    {
        Type sum = zero<Type>();
        for (auto mu : LORENTZ_INDICES)
        {
            sum += metric(mu) * left(mu) * right(mu);
        };
        return sum;
    };

    // Two vectors contracted with a projector
    template<class Type>
    inline Type contract(lorentz_tensor<Type,1> left, lorentz_tensor<Type, 2> P, lorentz_tensor<Type,1> right)
    {
        Type sum = zero<Type>();
        for (auto mu : LORENTZ_INDICES)
        {
            for (auto nu : LORENTZ_INDICES)
            {
                sum += metric({mu,nu}) * left(mu) * P(mu, nu) * right(nu);
            }
        };
        return sum;
    };

    // Two scalar rank-2's contracted 
    template<class Type>
    inline Type contract(lorentz_tensor<Type,2> left, lorentz_tensor<Type,2> right)
    {
        Type sum = zero<Type>();
        for (auto mu : LORENTZ_INDICES)
        {
            for (auto nu : LORENTZ_INDICES)
            {
                sum += metric(mu) * metric(nu) * contract(left(mu, nu), right(mu, nu));
            };
        };
        return sum;
    };

    // ---------------------------------------------------------------------------
    // Contractions involving the Levi-Civita tensor 

    // Return the value of the levi-civita symbol for some combination of lorentz_indices
    int levi_civita(lorentz_index mu, lorentz_index nu, lorentz_index alpha, lorentz_index beta);

    // Fully contract four vectors to a constant
    template<class T>
    inline T levi_civita(lorentz_tensor<T,1> q1, lorentz_tensor<T,1> q2, lorentz_tensor<T,1> q3, lorentz_tensor<T,1> q4)
    {
        T sum = zero<T>();
        for (auto mu : LORENTZ_INDICES)
        {
            for (auto alpha : LORENTZ_INDICES)
            {
                if (+mu - +alpha == 0) continue;
                for (auto beta : LORENTZ_INDICES)
                {
                    if ((+beta - +mu)*(+beta - +alpha) == 0) continue;
                    for (auto gamma : LORENTZ_INDICES)
                    {
                        int eps = levi_civita(mu, alpha, beta, gamma);
                        if (eps == 0) continue;

                        sum += eps * q1(mu) * q2(alpha) * q3(beta) * q4(gamma);
                    }
                }
            }
        }
        return sum;
    };

    
    // Contract three vectors to a vector
    template<class T>
    lorentz_tensor<T,1> levi_civita(lorentz_tensor<T,1> q1, lorentz_tensor<T,1> q2, lorentz_tensor<T,1> q3)
    {
        std::array<T,4> q;
        for (auto mu : LORENTZ_INDICES)
        {
            T sum = zero<T>();
            for (auto alpha : LORENTZ_INDICES)
            {
                if (+mu - +alpha == 0) continue;
                for (auto beta : LORENTZ_INDICES)
                {
                    if ((+beta - +mu)*(+beta - +alpha) == 0) continue;
                    for (auto gamma : LORENTZ_INDICES)
                    {
                        int eps = levi_civita(mu, alpha, beta, gamma);
                        if (eps == 0) continue;

                        sum += eps * q1(alpha) * q2(beta) * q3(gamma);
                    }
                }
            }
            q[+mu] = metric(mu) * sum;
        };

        return lorentz_vector(q);
    };
};

#endif