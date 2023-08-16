// Amplitude for production via a spin-1 exchange in the t-channel
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef COVARIANT_VECTOR_EXCHANGE_HPP
#define COVARIANT_VECTOR_EXCHANGE_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "form_factor.hpp"
#include "amplitude.hpp"

namespace jpacPhoto
{
    namespace covariant
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
                _covariants->update(helicities, s, t);

                complex result = contract(top_coupling(), propagator(), bottom_coupling());
                if (_option == kNoFF) return result;

                // Parse which argument should go into the form-factor
                // The exponential takes t' = t - tmin while monopole takes just t
                complex FF = (_option == kExpFF) ? _FF->eval(_t - _kinematics->t_min(s))
                                                 : _FF->eval(_t);

                // Multiply couplings with propagator
                return FF * result;
            };

            // Explicitly require s-channel helicities
            inline helicity_frame native_helicity_frame(){ return S_CHANNEL; };

            // We can have pseudo-scalar, vector, and axial-vector
            inline std::vector<particle> allowed_mesons(){  return { SCALAR, PSEUDOSCALAR, VECTOR, AXIALVECTOR }; };

            // And protons on the bottom vertex
            inline std::vector<particle> allowed_baryons(){ return { HALFPLUS }; };

            // The options here are the type of form_factor used
            // Default assumed exponential
            static const int kExpFF      = 0;
            static const int kMonopoleFF = 1;
            static const int kNoFF       = 2;
            inline void set_option( int opt )
            {
                switch (opt)
                {
                    case (kExpFF):      { _FF = new_FF<exponential>();  _option = kExpFF; break; }
                    case (kMonopoleFF): { _FF = new_FF<monopole>(_mEx); _option = opt;   break; }
                    case (kNoFF):       { _FF = nullptr; _option = opt; set_N_pars(3); return; }
                    default: { option_error(); return; };
                };

                // Before leaving make sure the new FF gets the last saved cutoff
                _FF->set_cutoff(_ffCutoff);
            };

            // Parameter names
            inline std::vector<std::string> parameter_labels()
            {
                std::vector<std::string> labels = { "gPhoton", "gN_Vector", "gN_Tensor"};
                if (_option != kNoFF)     labels.push_back("Cutoff");
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
            double _mEx;

            // Free parameters: two couplings and a form-factor cutoff
            double _gTop = 0, _gBotV = 0, _gBotT = 0, _ffCutoff = 0;

            // We include a t-channel form-factor for the propagator
            // By default this is the exponential one
            form_factor _FF = new_FF<exponential>();

            // Top coupling refers to the beam-exchange-meson interaction
            inline lorentz_tensor<complex,1> top_coupling()
            {
                // Beam
                auto q     = _covariants->q();
                auto eps   = _covariants->eps();
                
                // Outgoing meson
                auto q_p   = _covariants->q_prime();
                auto eps_p = _covariants->eps_prime();

                // Coupling function depends on
                // the quantum numbers of the produced meson
                lorentz_tensor<complex,1> T;
                switch ( _kinematics->get_meson() )
                {
                    case (AXIALVECTOR):
                    {
                        T = levi_civita(q, eps, eps_p); 
                        break;
                    };

                    default: { return lorentz_vector<complex>({0,0,0,0}); };
                };
                
                return _gTop * T;
            };

            // Bottom coupling refers to the target-exchange-recoil interation
            inline lorentz_tensor<complex,1> bottom_coupling()
            {
                auto u     = _covariants->u();    // Target spinor
                auto ubar  = _covariants->ubar(); // Recoil spinor
                auto k     = _covariants->k_t();  // t-channel exchange momentum

                // This is also dependent on baryon quantum numbers
                // We have two pieces for the vector and tensor currents
                lorentz_tensor<complex,1> vector, tensor;
                switch ( _kinematics->get_baryon() )
                {
                    case (HALFPLUS):
                    {
                        auto vector_current =  gamma_vector();
                        auto tensor_current = (gamma_vector()*slash(k) - slash(k)* gamma_vector())/2;

                        vector = bilinear(ubar, vector_current, u); 
                        tensor = bilinear(ubar, tensor_current, u);
                        break;
                    }
                    
                    default: return lorentz_vector<complex>({NaN<complex>(),NaN<complex>(),NaN<complex>(),NaN<complex>()});  break;
                };  

                return is_zero(_gBotT) ? _gBotV*vector : _gBotV*vector - (_gBotT/(_mT+_mR))*tensor; 
            };

            // Spin-1 propagator in the t-channel
            inline lorentz_tensor<complex,2> propagator()
            {
                auto k = _covariants->k_t();
                auto projector = metric_tensor() - tensor_product(k, k)/_t;
                return  -I*projector/(_t - _mEx*_mEx);
            };
            
        };
    };
};

#endif