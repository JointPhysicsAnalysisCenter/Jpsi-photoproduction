// Custom implementation of Lorentz tensors.
// These are assembled by the outer/tensor products of vectors and metrics
// the two "fundamenetal" Lorentz tensors. The third structure (Levi-Civita) is defined
// as a function instead
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@iu.edu
// ------------------------------------------------------------------------------

#ifndef LORENTZ_TENSOR_HPP
#define LORENTZ_TENSOR_HPP

#include <cstddef>
#include <vector>
#include <memory>

#include "constants.hpp"
#include "tensor_object.hpp"

namespace jpacPhoto
{

    template<class Type, int Rank>
    class lorentz_tensor : public tensor_object<Type>
    {
        // The shape is always a Rank x 4 tensor. 
        // The Type depends on the possibility of dirac structures
        // Scalar (complex/double) quantities is any object with only Lorentz structure
        // dirac_spinor and dirac_matrix tensors allow constructor of high half-integer spin wavefunctions

        public: 

        // Default tensor with nothing initialized
        lorentz_tensor<Type,Rank>(){};

        // Copy constructor
        lorentz_tensor<Type,Rank>(lorentz_tensor<Type,Rank> const & old)
        : _lhsN(old._lhsN), _rhsN(old._rhsN), 
          _conj(old._conj), 
          _subtensors(old._subtensors),
          _is_sum(old._is_sum)
        {};

        // Implicit constructor, stores pointers to constituent tensors of smaller rank
        lorentz_tensor<Type,Rank>(std::vector<std::shared_ptr<tensor_object<Type>>> Ts, bool sum)
        : _subtensors(Ts), _is_sum(sum)
        {};

        // Get the rank of the tensor (number of open indicies)
        const inline int rank(){ return Rank; };

        // Re-assignment, just makes a copy basically
        inline lorentz_tensor<Type,Rank> & operator=(lorentz_tensor<Type,Rank> const & T)
        {
            _lhsN           = T._lhsN;
            _rhsN           = T._rhsN;
            _conj           = T._conj;
            _subtensors     = T._subtensors;
            _is_sum         = T._is_sum;
            return *this;
        };

        // -----------------------------------------------------------------------
        // Evaluate elements of the tensor

        inline virtual Type operator()(std::vector<lorentz_index> indices)
        {
            if (indices.size() != Rank) return error("lorentz_tensor", "Incorrect number of indices passed!", NaN<Type>());
            if (_is_sum)
            {
                Type sum = zero<Type>();
                for (auto T : _subtensors)
                {
                    sum += T->operator()(indices);
                };

                Type result = _lhsN*sum*_rhsN;
                return (_conj) ? conj(result) : result;
            };
            
            auto running_index = indices.begin();
            int subrank0 = _subtensors[0]->rank();
            std::vector<lorentz_index> ind0 = std::vector<lorentz_index>(running_index, running_index + subrank0);

            Type prod = _subtensors[0]->operator()(ind0);
            running_index += subrank0;

            for (int i = 1; i < _subtensors.size(); i++)
            {
                int subrank = _subtensors[i]->rank();
                prod *= _subtensors[i]->operator()( std::vector<lorentz_index>(running_index, running_index + subrank));
                running_index += subrank;
            };
            Type result = _lhsN*prod*_rhsN;
            return (_conj) ? conj(result) : result; 
        };
        inline Type operator()(lorentz_index mu){ return operator()({{mu}}); };
        inline Type operator()(lorentz_index mu, lorentz_index nu){ return operator()({{mu,nu}});};

        // -----------------------------------------------------------------------
        // Unary operations 

        // Multiply and divide by a constant, assumed on the right
        inline lorentz_tensor<Type,Rank> & operator*=(Type c)
        {
            // Apply the constant to the first index simply
            _rhsN *= c*identity<Type>();
            return *this;
        };
        inline lorentz_tensor<Type,Rank> & operator/=(complex c)
        {
            _rhsN *= (1/c)*identity<Type>();
            return *this;
        };

        // Negate a tensor
        inline lorentz_tensor<Type,Rank> operator-()
        {
            lorentz_tensor<Type,Rank> neg = *this;
            neg._lhsN *= -1*identity<Type>();
            return neg;
        };

        // Return a copy of itself with the conj flag flipped
        inline lorentz_tensor<Type,Rank> conjugate()
        {
            lorentz_tensor<Type,Rank> copy = *this;
            copy._conj = !_conj; 
            return copy;
        };

        protected:

        // Multiply tensors by non-tensor objects
        template<class T, int R>
        friend lorentz_tensor<T,R> operator*(T c, lorentz_tensor<T,R> rhs);
        template<class T, int R>
        friend lorentz_tensor<T,R> operator*(lorentz_tensor<T,R> lhs, T c);

        // "Capture" constructor require more access than other contructors
        template<int R>
        friend lorentz_tensor<dirac_matrix,R> operator*(dirac_matrix c, lorentz_tensor<complex,R> rhs);
        template<int R>
        friend lorentz_tensor<dirac_matrix,R> operator*(lorentz_tensor<complex,R> lhs, dirac_matrix c);
        template<int R>
        friend lorentz_tensor<dirac_spinor,R> operator*(dirac_spinor c, lorentz_tensor<complex,R> rhs);

        // Tensor dirac_spinor * matrices
        template<int R>
        friend lorentz_tensor<dirac_spinor,R> operator*(dirac_matrix M, lorentz_tensor<dirac_spinor,R> rhs);
        template<int R>
        friend lorentz_tensor<dirac_spinor,R> operator*(lorentz_tensor<dirac_spinor,R> lhs, dirac_matrix M);

        // Whether or not to take the conjugate of the element
        bool _conj = false;

        // If not, we the tensor serves as a store of constituent other tensors 
        // which are summed element wise instead
        bool _is_sum = false;

        // If its a more complicated object (rank >= 3) we store pointers to its component tensors
        std::vector<std::shared_ptr<tensor_object<Type>>> _subtensors;

        // If the tensor is a metric, it may still be multiplied by a constant.
        // This however may be generalized depending on the substructure inside the tensor
        Type _lhsN = identity<Type>(), _rhsN = identity<Type>();

        // Both type and rank has to be the same to add
        inline void add_tensor(lorentz_tensor<Type,Rank> T)
        {
            if (!_is_sum)
            {
                warning("add_tensor()", "Cannot add tensor to pre-initialized one. Initilize a new tensor as the sum!");
                return;
            };
            _subtensors.push_back(std::make_shared<lorentz_tensor<Type,Rank>>(T)); 
        };

        // Return a dirac_spinor version of complex tensor
        inline std::shared_ptr<tensor_object<dirac_spinor>> spinorify()
        {
            dirac_spinor id = identity<dirac_spinor>();

            // Else go iteratively through sum terms first
            if (_is_sum)
            {
                std::vector<std::shared_ptr<tensor_object<dirac_spinor>>> new_to_sum;
                for (auto term : _subtensors)
                {
                    new_to_sum.push_back(term->spinorify());
                }
                // True in constructor means the subtesnors are summed not multipled
                return std::make_shared<lorentz_tensor<dirac_spinor,Rank>>(new_to_sum, true);
            };

            // If we reach this point it means we have a tensor product on our hands,
            // similar to above we simply go through each subtensor and spinorify it in turn
            std::vector<std::shared_ptr<tensor_object<dirac_spinor>>> new_subtensors;
            for (auto term : _subtensors)
            {
                new_subtensors.push_back(term->spinorify());
            }

            // Unlike above, the false means multipled
            return std::make_shared<lorentz_tensor<dirac_spinor,Rank>>(new_subtensors, false);
        };

        // Return a dirac_matrix
        // This creates a problem if mixing dirac_matrices and spinors 
        inline std::shared_ptr<tensor_object<dirac_matrix>> matrixify()
        {    
            // Else go iteratively through sum terms first
            if (_is_sum)
            {
                std::vector<std::shared_ptr<tensor_object<dirac_matrix>>> new_to_sum;
                for (auto term : _subtensors)
                {
                    new_to_sum.push_back(term->matrixify());
                }
                // True in constructor means the subtesnors are summed not multipled
                return std::make_shared<lorentz_tensor<dirac_matrix,Rank>>(new_to_sum, true);
            };

            // If we reach this point it means we have a tensor product on our hands,
            // similar to above we simply go through each subtensor and matrixify it in turn
            std::vector<std::shared_ptr<tensor_object<dirac_matrix>>> new_subtensors;
            for (auto term : _subtensors)
            {
                new_subtensors.push_back(term->matrixify());
            }

            // Unlike above, the false means multipled
            return std::make_shared<lorentz_tensor<dirac_matrix,Rank>>(new_subtensors, false);
        };
    };

    // ---------------------------------------------------------------------------
    // NON-MEMBER FUNCTIONS

    // ---------------------------------------------------------------------------
    // "Constructor" functions 

    // "Constructor" of an arbitrary rank-1 tensor (i.e. vector)
    template<class Type = complex>
    inline lorentz_tensor<Type,1> lorentz_vector(std::array<Type,4> entries)
    {
        auto ptr = std::make_shared<raw_lorentz_vector<Type>>(entries);
        return lorentz_tensor<Type,1>({ptr}, false);
    };

    // The metric tensor is unique rank-2 tensor
    //  as it cannot be decomposed into a tensor product of rank-1 vectors
    template<class Type = complex>
    inline lorentz_tensor<Type,2> metric_tensor()
    {
        std::shared_ptr<tensor_object<Type>> ptr = std::make_shared<raw_metric_tensor<Type>>();
        return lorentz_tensor<Type,2>({ptr}, false);
    };

    // N vectors -> rank-N tensor
    template<class T = complex, int R>
    inline lorentz_tensor<T,R> tensor_product(std::array<lorentz_tensor<T,1>,R> vs)
    {
        std::vector< std::shared_ptr< tensor_object<T> > > ptrs;
        for (auto vec : vs)
        {
            ptrs.push_back(std::make_shared<lorentz_tensor<T,1>>(vec));
        };

        return lorentz_tensor<T,R>(ptrs, false);
    };

    // Generic tensor products for to create abritrary sized tensors
    template<class Type = complex, int R1, int R2> 
    inline lorentz_tensor<Type,R1+R2> tensor_product(lorentz_tensor<Type,R1> lhs, lorentz_tensor<Type,R2> rhs)
    {
        std::vector< std::shared_ptr< tensor_object<Type> > > ptrs;
        ptrs.push_back( std::make_shared<lorentz_tensor<Type,R1>>(lhs) );
        ptrs.push_back( std::make_shared<lorentz_tensor<Type,R2>>(rhs) );

        return lorentz_tensor<Type, R1+R2>(ptrs, false);
    };

    // ---------------------------------------------------------------------------
    // Scalar tensors "capture" dirac data types they get multipled by a non-scalar dirac type
    // We dont use template here because compiler gets confused between complex, double, int, etc.
    // We want to always default to complex even if we write p = -2*q;

    // complex * dirac_matrix -> dirac_matrix
    template<int R>
    inline lorentz_tensor<dirac_matrix,R> operator*(dirac_matrix c, lorentz_tensor<complex,R> rhs)
    {
        auto matrified = rhs.matrixify();
        return c * static_cast<lorentz_tensor<dirac_matrix,R>&>(*matrified);
    };
    template<int R>
    inline lorentz_tensor<dirac_matrix,R> operator*(lorentz_tensor<complex,R> lhs, dirac_matrix c)
    {  
        auto matrified = lhs.matrixify();
        return static_cast<lorentz_tensor<dirac_matrix,R>&>(*matrified) * c;
    };

    // complex * dirac_spinor -> dirac_spinor
    template<int R>
    inline lorentz_tensor<dirac_spinor,R> operator*(dirac_spinor c, lorentz_tensor<complex,R> rhs)
    {
        auto matrified = rhs.spinorify();
        return c * static_cast<lorentz_tensor<dirac_spinor,R>&>(*matrified);
    };
    template<int R>
    inline lorentz_tensor<dirac_spinor,R> operator*(lorentz_tensor<complex,R> lhs, dirac_spinor c)
    {  return c * lhs; };
    
    // --------------------------------------------------------------------------
    // Interactions with constant scalar types 

    // Multiplication by the same type
    // Order matters here only becasue of possible matrix multipication
    template<class T = complex, int R>
    inline lorentz_tensor<T,R> operator*(T c, lorentz_tensor<T,R> rhs)
    {
        rhs._lhsN = c * rhs._lhsN;
        return rhs;
    };
    template<class T = complex, int R>
    inline lorentz_tensor<T,R> operator*(lorentz_tensor<T,R> lhs, T c)
    {
        lhs._rhsN = lhs._rhsN * c;
        return lhs;
    };

    // Multiplication by an integer type, order no longer matters

    // int
    template<class T = complex, int R>
    inline lorentz_tensor<T,R> operator*(int c, lorentz_tensor<T,R> rhs)
    { return (c*identity<T>())*rhs; };
    template<class T = complex, int R>
    inline lorentz_tensor<T,R> operator*(lorentz_tensor<T,R> lhs, int c)
    { return c*lhs; };

    // double
    template<class T = complex, int R>
    inline lorentz_tensor<T,R> operator*(double c, lorentz_tensor<T,R> rhs)
    { return (c*identity<T>())*rhs; };
    template<class T = complex, int R>
    inline lorentz_tensor<T,R> operator*(lorentz_tensor<T,R> lhs, double c)
    { return c*lhs; };
    
    
    // we need to define complex * dirac_spinor and dirac_matrix seperately becasue else the template above gets confused
    template<int R>
    inline lorentz_tensor<dirac_spinor,R> operator*(complex c, lorentz_tensor<dirac_spinor,R> rhs)
    { return (c*identity<dirac_spinor>())*rhs; };
    template<int R>
    inline lorentz_tensor<dirac_spinor,R> operator*(lorentz_tensor<dirac_spinor,R> lhs, complex c)
    { return (c*identity<dirac_spinor>())*lhs; }; 

    template<int R>
    inline lorentz_tensor<dirac_matrix,R> operator*(complex c, lorentz_tensor<dirac_matrix,R> rhs)
    { return (c*identity<dirac_matrix>())*rhs; };
    template<int R>
    inline lorentz_tensor<dirac_matrix,R> operator*(lorentz_tensor<dirac_matrix,R> lhs, complex c)
    { return (c*identity<dirac_matrix>())*lhs; }; 

    // Division only makes sense for dirac scalar types
    template<class LType, class RType, int R>
    inline lorentz_tensor<LType,R> operator/(lorentz_tensor<LType,R> lhs, RType c){ return (1./complex(c)) * lhs; };
    
    // --------------------------------------------------------------------------
    // Interactions with constant non-scalar types 

    // Multiplying matrices and spinors with tensor structure
    template<int R>
    inline lorentz_tensor<dirac_spinor,R> operator*(dirac_matrix M, lorentz_tensor<dirac_spinor,R> rhs)
    {
        rhs._lhsN = M * rhs._lhsN;
        return rhs;
    };
    template<int R>
    inline lorentz_tensor<dirac_spinor,R> operator*(lorentz_tensor<dirac_spinor,R> lhs, dirac_matrix M)
    {
        lhs._rhsN = lhs._rhsN * M;
        return lhs;
    };

    // ---------------------------------------------------------------------------
    // Operations between tensors

    // Add two vectors together.
    // For arbitrary size, this is done by saving tensors and adding iteratively through terms in the sum
    // when an element is asked for
    template<class T, int R>
    inline lorentz_tensor<T,R> operator+(lorentz_tensor<T,R> lhs, lorentz_tensor<T,R> rhs)
    {
        auto l = std::make_shared<lorentz_tensor<T,R>>(lhs);
        auto r = std::make_shared<lorentz_tensor<T,R>>(rhs);
        lorentz_tensor<T,R> sum({l,r}, true);
        return sum;
    };
    
    // Subtract two vectors together
    template<class T, int R>
    inline lorentz_tensor<T,R> operator-(lorentz_tensor<T,R> lhs, lorentz_tensor<T,R> rhs)
    {
        return lhs + (-rhs);
    };
};

#endif