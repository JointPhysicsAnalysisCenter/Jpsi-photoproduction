// Special version of lorentz_tensor which corresponds to a metric tensor
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------

#ifndef METRIC_HPP
#define METRIC_HPP

#include "constants.hpp"
#include "dirac_spinor.hpp"
#include "dirac_matrix.hpp"

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // Lorentz index handling

    // Enclosing lorentz indices in an enum makes summing over them more transparent
    // Also has the benefit of not needing to check type or scope of a given index
    enum class lorentz_index: int {t = 0, x = 1, y = 2, z = 3};
    const std::array<lorentz_index,4> LORENTZ_INDICES = {lorentz_index::t, lorentz_index::x, lorentz_index::y, lorentz_index::z};
    
    // e.g. int x = +lorentz_index(mu);
    inline constexpr unsigned operator+(lorentz_index x)
    {
        return static_cast<unsigned>(x);
    };

    // This allows lorentz_indices to be cout'ed
    inline std::ostream &operator<<(std::ostream &os, lorentz_index mu)
    { 
        return os << +mu;
    };

    // ---------------------------------------------------------------------------
    // This function is only to make the "capture" constructors below to work

    template<class In, class Out>
    inline auto capture(In x)
    {
        return identity<Out>() * x;
    };

    // The above is the behavior we want in most cases except the
    // following two exceptions which are errors
    template<>
    inline auto capture<dirac_matrix,dirac_spinor>(dirac_matrix x)
    {
        return NaN<dirac_spinor>();
    };
    template<>
    inline auto capture<dirac_spinor,dirac_matrix>(dirac_spinor x)
    {
        return NaN<dirac_matrix>();
    };

    // -----------------------------------------------------------------------
    // In most useful contexts, these come from outer products of vectors 
    // This class describes an arbitrary rank tensor created iteratively 

    template<class Type, int Rank>
    class lorentz_tensor;

    template<class Type = complex>
    class tensor_object 
    {
        public: 
        virtual const int rank() = 0;
        virtual Type operator()(std::vector<lorentz_index> indices) = 0;

        virtual std::shared_ptr<tensor_object<dirac_matrix>> matrixify(){ return nullptr; };
        virtual std::shared_ptr<tensor_object<dirac_spinor>> spinorify(){ return nullptr; };
    };

    // -----------------------------------------------------------------------
    // Lorentz vectors are simple enough we can store them elementwise
    template<class Type>
    class raw_lorentz_vector : public tensor_object<Type>
    {
        public: 

        // Default constructor
        raw_lorentz_vector(std::array<Type,4> entries)
        : _entries(entries)
        {};

        // Always a rank 1 tensor
        inline const int rank(){ return 1; };

        inline Type operator()(std::vector<lorentz_index> indices)
        {
            if (indices.size() != rank()) return error("lorentz_tensor", "Incorrect number of indices passed!", NaN<Type>());
            int mu = +indices[0];
            return _entries[mu];
        };

        // Elements stored locally
        std::array<Type,4> _entries;

        inline std::shared_ptr<tensor_object<dirac_matrix>> matrixify()
        { 
            std::array<dirac_matrix,4> new_entries;
            for (auto mu : LORENTZ_INDICES)
            {
                new_entries[+mu] = capture<Type,dirac_matrix>(_entries[+mu]);
            };
            return std::make_shared<raw_lorentz_vector<dirac_matrix>>(new_entries);
        };     
        inline std::shared_ptr<tensor_object<dirac_spinor>> spinorify()
        { 
            std::array<dirac_spinor,4> new_entries;
            for (auto mu : LORENTZ_INDICES)
            {
                new_entries[+mu] = capture<Type,dirac_spinor>(_entries[+mu]);
            };
            return std::make_shared<raw_lorentz_vector<dirac_spinor>>(new_entries);
        };
    };

    // -----------------------------------------------------------------------
    // Only non factorizable rank 2 tensor, the metric

    template<class Type>
    class raw_metric_tensor : public tensor_object<Type>
    {
        public: 

        // Default constructor
        raw_metric_tensor(){};

        // Always a rank 2 tensor
        inline const int rank(){ return 2; };

        inline Type operator()(std::vector<lorentz_index> indices)
        {
            if (indices.size() != rank()) return error("lorentz_tensor", "Incorrect number of indices passed!", NaN<Type>());
            int mu = +indices[0], nu = +indices[1];
            return (mu == nu) ? complex((mu == 0) - (mu != 0)) * id : z;
        };

        // Elements are zero or 1
        Type z = zero<Type>(), id = identity<Type>();

        inline std::shared_ptr<tensor_object<dirac_matrix>> matrixify()
        { 
            return std::make_shared<raw_metric_tensor<dirac_matrix>>();
        };        
        inline std::shared_ptr<tensor_object<dirac_spinor>> spinorify()
        { 
            return std::make_shared<raw_metric_tensor<dirac_spinor>>();
        };
    };

    // -----------------------------------------------------------------------
    // Only non factorizable rank 4 tensor, the levi-civita

    template<class Type>
    class raw_levicivita_tensor : public tensor_object<Type>
    {
        public: 

        // Default constructor
        raw_levicivita_tensor(){};

        // Always a rank 2 tensor
        inline const int rank(){ return 4; };

        inline Type operator()(std::vector<lorentz_index> indices)
        {
            if (indices.size() != rank()) return error("lorentz_tensor", "Incorrect number of indices passed!", NaN<Type>());

            // Convert indices to their ints
            int a = +indices[0], b = +indices[1], c = +indices[2], d = +indices[3];
            int eps = (d - c) * (d - b) * (d - a) * (c - b) * (c - a) * (b - a);
            complex result = (eps == 0) ? eps : eps / abs(eps);

            return result * identity<Type>();
        };

        inline std::shared_ptr<tensor_object<dirac_matrix>> matrixify()
        { 
            return std::make_shared<raw_levicivita_tensor<dirac_matrix>>();
        };        
        inline std::shared_ptr<tensor_object<dirac_spinor>> spinorify()
        { 
            return std::make_shared<raw_levicivita_tensor<dirac_spinor>>();
        };
    };

};

#endif