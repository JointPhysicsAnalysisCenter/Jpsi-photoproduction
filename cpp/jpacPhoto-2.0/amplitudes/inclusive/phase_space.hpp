//
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------

#ifndef PHASESPACE_HPP       
#define PHASESPACE_HPP

#include "constants.hpp"
#include "inclusive_process.hpp"

namespace jpacPhoto
{
    namespace 
    {
        class phase_space : public raw_inclusive_process
        {
            public: 

            phase_space(inclusive_key key, double mX, double mMin, std::string id = "")
            : raw_inclusive_process(key, mX, id), _mMin2(mMin*mMin)
            {
                set_N_pars(0);
            };

            // Minimum mass is the proton 
            double minimum_M2(){ return _mMin2; };

            // Amplitude is a constant
            double invariant_xsection(double s, double t, double mm){ return 1; };

            private:

            // Set the minimum mass
            double _mMin2 = (M_PROTON + M_PION)*(M_PROTON + M_PION);
        };
    };
};

#endif