// Abstract class to represent a total hadronic cross-section
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------

#ifndef TOTAL_XSECTION_HPP
#define TOTAL_XSECTION_HPP

#include <memory>
#include "constants.hpp"

namespace jpacPhoto
{
    class raw_total_xsection;
    using total_xsection = std::shared_ptr<raw_total_xsection>;

    template<class T>
    inline total_xsection new_total_xsection()
    {
        total_xsection ptr = std::make_shared<T>();
        return ptr;
    };

    template<class T, class A>
    inline total_xsection new_total_xsection(A par)
    {
        total_xsection ptr = std::make_shared<T>(par);
        return ptr;
    };

    template<class T, class A, class B>
    inline total_xsection new_total_xsection(A par1, B par2)
    {
        total_xsection ptr = std::make_shared<T>(par1, par2);
        return ptr;
    };

    template<class T, class A, class B, class C>
    inline total_xsection new_total_xsection(A par1, B par2, C par3)
    {
        total_xsection ptr = std::make_shared<T>(par1, par2, par3);
        return ptr;
    };

    class raw_total_xsection 
    {
        public: 
        
        raw_total_xsection(){};

        raw_total_xsection(std::array<double, 2> m)
        : _mB(m[0]), _mT(m[1])
        {};

        // Just need to define how to evaluate the cross-section
        // We may allow an additional variable q2 to consider virtuality effects 
        virtual double evaluate(double s, double q2) = 0;
        inline  double evaluate(double s){ return evaluate(s, 0); };

        // Mass of beam and target
        double _mB = 0, _mT = 0;
    };
};

#endif