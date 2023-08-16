// Extention of the analytic spin-0 exchange to a reggeized psuedoscalar in the t-channel
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef REGGE_PSEUDOSCALAR_EXCHANGE_HPP
#define REGGE_PSEUDOSCALAR_EXCHANGE_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "amplitude.hpp"
#include "cgamma.hpp"
#include "analytic/pseudoscalar_exchange.hpp"

namespace jpacPhoto
{
    namespace regge
    {
        class pseudoscalar_exchange : public analytic::pseudoscalar_exchange
        {
            public: 

            // Constructor calls the analytic one with no exchange mass
            pseudoscalar_exchange(amplitude_key key, kinematics xkinem, std::string id)
            : analytic::pseudoscalar_exchange(key, xkinem, 0, id)
            {
                // Two additional parameters for regge trajectory
                initialize(5);
            };

            inline complex helicity_amplitude(std::array<int, 4> helicities, double s, double t)
            {
                // Save inputs
                store(helicities, s, t);
                _qi = _kinematics->initial_momentum_tframe(_t);
                _qf = _kinematics->final_momentum_tframe(_t);

                // An on-shell spin-0 exchange cannot flip helicities
                bool helicity_conserving = (_lamB == _lamX) && (_lamR == _lamT);
                if (!helicity_conserving) return 0;

                complex result = top_coupling() * propagator() * bottom_coupling();
                if (_option == kNoFF) return result;

                // Parse which argument should go into the form-factor
                // The exponential takes t' = t - tmin while monopole takes just t
                complex FF = (_option == kExpFF) ? _FF->eval(_t - _kinematics->t_min(s))
                                                : _FF->eval(_t);

                // Multiply couplings with propagator
                return FF * result;
            };

            // -----------------------------------------------------------------------
            // Internal data members 

            protected:

            // Have five free parameters: 
            // [0] Top (beam-exchange-meson) coupling
            // [1] Bottom (target-exchange-recoil) coupling
            // [2] Form-factor cutoff
            // [3] Trajectory intercept
            // [4] trajectory slope
            inline void allocate_parameters(std::vector<double> x)
            {
                _gTop     = x[0];
                _gBot     = x[1];
                _ffCutoff = x[2];
                _a0       = x[3];
                _aP       = x[4];

                // Pass cutoff to FF as well
                _FF->set_cutoff(_ffCutoff);
                return;
            };

            // Intercept and slope of exchange trajectory
            double _a0 = 0, _aP = 0;

            // The only thing that changes is the propagator which now takes the regge pole form
            inline complex propagator()
            {
                double alpha = _a0 + _aP * _t; // linear trajectory
                if (std::abs(alpha) > 20) return 0;
                
                std::complex<double> signature_factor = (1 + exp(-I*PI*alpha))/2;
                return signature_factor * cgamma(-alpha) * pow(_s, alpha);
            };  
        };
    };
};

#endif