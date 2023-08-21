// Amplitude defining production via a spin-1 exchange in the t-channel
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef ANALYTIC_VECTOR_EXCHANGE_HPP
#define ANALYTIC_VECTOR_EXCHANGE_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "form_factor.hpp"
#include "amplitude.hpp"

namespace jpacPhoto
{
    namespace analytic
    {
        class vector_exchange : public raw_amplitude
        {
            public:

            // Constructor
            vector_exchange(amplitude_key key, kinematics xkinem, double exchange_mass, std::string id)
            : raw_amplitude(key, xkinem, id),
              _mEx(exchange_mass)
            {
                initialize(4);
            };

            // ---------------------------------------------------------------------------
            // Defining the virtual functions required of an amplitude
            
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
                
                // Double flip is forbidden
                if (abs(_lam) == 2) return 0;
              
                complex result = top_coupling() * propagator() * bottom_coupling();
                if (_option == kNoFF) return result;

                // Parse which argument should go into the form-factor
                // The exponential takes t' = t - tmin while monopole takes just t
                complex FF = (_option == kExpFF) ? _FF->eval(_t - _kinematics->t_min(s))
                                                 : _FF->eval(_t);

                // Multiply couplings with propagator
                return FF * result;
            };

            // Explicitly require t-channel helicities
            inline helicity_frame native_helicity_frame(){ return T_CHANNEL; };

            // We can have pseudo-scalar, vector, and axial-vector
            inline std::vector<particle> allowed_mesons()
            {
                return { PSEUDOSCALAR, VECTOR, AXIALVECTOR };
            };

            // But only either parity spin-1/2
            inline std::vector<particle> allowed_baryons()
            {
                return { HALFPLUS };
            };


            // The options here are the type of form_factor used
            static const int kExpFF      = 0;
            static const int kMonopoleFF = 1;
            static const int kNoFF       = 2;
            inline void set_option( int opt )
            {
                switch (opt)
                {
                    case (kExpFF): 
                    {
                        _FF = new_FF<exponential>();
                        _option = opt;
                        break;
                    };
                    case (kMonopoleFF):
                    {
                        _FF = new_FF<monopole>(_mEx);
                        _option = opt;
                        break;
                    }
                    case (kNoFF):
                    {
                        _FF = nullptr;
                        _option = opt;
                        set_N_pars(3);
                        return;
                    }
                    default: 
                    {
                        option_error();
                        return;
                    };
                };

                // Before leaving make sure the new FF gets the last saved cutoff
                _FF->set_cutoff(_ffCutoff);
            };

            // Parameter names
            inline std::vector<std::string> parameter_labels()
            {
                std::vector<std::string> labels = { "gPhoton", "gN_Vector", "gN_Tensor"};
                if (_option != kNoFF)    labels.push_back("Cutoff");
                return labels;
            };
            
            // -----------------------------------------------------------------------
            // Internal data members 

            protected:

            // Have three free parameters: 
            // [0] Top (beam-exchange-meson) coupling
            // [1] Bottom (target-exchange-recoil) coupling
            // [2] Form-factor cutoff
            inline void allocate_parameters(std::vector<double> x)
            {
                _gTop     = x[0];
                _gBotV    = x[1];
                _gBotT    = x[2];

                if (_option != kNoFF)
                {
                    _ffCutoff = x[3];
                    _FF->set_cutoff(_ffCutoff);
                };
                return;
            };

            // Exchange mass
            double _mEx, _zt;

            // Free parameters: two couplings and a form-factor cutoff
            double _gTop = 0, _gBotV = 0, _gBotT = 0, _ffCutoff = 0;

            // Net helicities
            int _lam = 0, _lamp = 0, _M = 0;

            // Initial & final momentum in the t-channel frame
            complex _qi = 0, _qf = 0;

            // We include a t-channel form-factor for the propagator
            // By default this is the exponential one
            form_factor _FF = new_FF<exponential>();

            // Top coupling refers to the beam-exchange-meson interaction
            inline complex top_coupling()
            {
                // Coupling function depends on
                // the quantum numbers of the produced meson
       
                complex result = 0;
                switch ( _kinematics->get_meson() )
                {
                    case (AXIALVECTOR):  result = (abs(_lam) == 0) ? 1 : csqrt(_t) / _mX; break;
                    
                    case (VECTOR):       result = (abs(_lam) == 0) ? 1 + (1-abs(_lamB)*_mB/_mX) : (- csqrt(_t)/_mX); break;

                    case (PSEUDOSCALAR): result = (_kinematics->is_photon()) ? -4*csqrt(_t) : csqrt(_t); break;

                    default: break;
                };

                return _gTop*_qi*result;
            };

            // Bottom coupling refers to the target-exchange-recoil interation
            inline complex bottom_coupling()
            {
                // This is also dependent on baryon quantum numbers
                // We have two pieces for the vector and tensor currents
                complex vector = 0, tensor = 0;
                switch ( _kinematics->get_baryon() )
                {
                    case (HALFPLUS):
                    {
                        vector = (abs(_lamp) == 0) ? _mT + _mR : csqrt(2*_t);
                        tensor = (abs(_lamp) == 0) ? _t/(_mT+_mR) : csqrt(2*_t);
                    };
                    default: break;
                };

                return _lamT*(_gBotV * vector + _gBotT * tensor)*csqrt(1 - pow(_mT-_mR,2)/_t);
            };

            // Spin-1 propagator in the t-channel
            inline complex propagator()
            {
                return - I * wigner_d_int_cos(1, _lam, _lamp, _zt) / (_t - _mEx*_mEx);
            };
            
        };
    };
};

#endif