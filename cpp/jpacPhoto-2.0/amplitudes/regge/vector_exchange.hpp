// Extention of the analytic spin-1 exchange to a reggeized vectors in the t-channel
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef REGGE_VECTOR_EXCHANGE_HPP
#define REGGE_VECTOR_EXCHANGE_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "amplitude.hpp"
#include "cgamma.hpp"
#include "analytic/vector_exchange.hpp"

namespace jpacPhoto
{
    namespace regge
    {
        class vector_exchange : public analytic::vector_exchange
        {
            public: 

            // Constructor calls the analytic one with no exchange mass
            vector_exchange(amplitude_key key, kinematics xkinem, std::string id)
            : analytic::vector_exchange(key, xkinem, 0, id)
            {
                // Two additional parameters for regge trajectory
                initialize(6);
            };

            inline complex helicity_amplitude(std::array<int, 4> helicities, double s, double t)
            {
                // Save inputs
                store(helicities, s, t);

                // t-channel kinematic quantities
                _qi = _kinematics->initial_momentum_tframe(_t);
                _qf = _kinematics->final_momentum_tframe(_t);
                _zt = _kinematics->z_t(_s, _theta);

                // Net helicities
                _lam  =  _lamB - _lamX;
                _lamp = (_lamT - _lamR) / 2;
                _M    = std::max(abs(_lam), abs(_lamp));

                // Double flip is forbidden
                if (_M == 2) return 0;
                
                // Parse which argument should go into the form-factor
                // The exponential takes t' = t - tmin while monopole takes just t
                complex FF = (_option == kExpFF) ? _FF->eval(_t - _kinematics->t_min(s))
                                                 : _FF->eval(_t);

                // Multiply couplings with propagator
                return FF * top_coupling() * propagator() * bottom_coupling();
            };

            // Parameter names
            inline std::vector<std::string> parameter_labels()
            {
                return { "gPhoton", "gN_Vector", "gN_Tensor", "Intercept", "Slope"};
            };

            // -----------------------------------------------------------------------
            // Internal data members 

            protected:

            // Have five free parameters: 
            // [0] Top (beam-exchange-meson) coupling
            // [1] Bottom Vector coupling
            // [2] Bottom Tensor coupling
            // [3] Form-factor cutoff
            // [4] Trajectory intercept
            // [5] trajectory slope
            inline void allocate_parameters(std::vector<double> x)
            {
                _gTop     = x[0];
                _gBotV    = x[1];
                _gBotT    = x[2];
                _ffCutoff = x[3];
                _inter    = x[4];
                _slope    = x[5];

                // Pass cutoff to FF as well
                _FF->set_cutoff(_ffCutoff);
                return;
            };

            // Intercept and slope of exchange trajectory
            double _inter = 0, _slope = 0;

            // Maximal helicity flip
            int _M = 0;

            // The only thing that changes is the propagator which now takes the regge pole form
            inline complex propagator()
            {
                double alpha = _inter + _slope * _t; // linear trajectory
                if (std::abs(alpha) > 30) return 0;
                
                complex norm = - wigner_leading_coeff(1, _lam, _lamp) * _slope;

                complex signature_factor  = (-1 + exp(-I*PI*alpha))/2;

                complex half_angle_factor = pow( csqrt((1-_zt)/2), abs(_lam - _lamp)) 
                                          * pow( csqrt((1+_zt)/2), abs(_lam + _lamp));

                complex barrier_factor = pow(2*_qi*_qf, 1 - _M);

                return norm * signature_factor * half_angle_factor / barrier_factor * cgamma(1-alpha) * pow(_s, alpha - _M);
            };  
        };
    };
};

#endif