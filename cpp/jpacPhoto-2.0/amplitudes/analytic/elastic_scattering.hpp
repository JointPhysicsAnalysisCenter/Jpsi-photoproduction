// Implementation of a PWA assuming single-channel elastic scattering in K-matrix
// formalism
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------

#ifndef ANALYTIC_ELASTIC_SCATTERING
#define ANALYTIC_ELASTIC_SCATTERING

#include "constants.hpp"
#include "kinematics.hpp"
#include "amplitude.hpp"

namespace jpacPhoto
{
    namespace analytic
    {
        class elastic_scattering : public raw_partial_wave
        {
            public: 

            elastic_scattering(amplitude_key key, kinematics xkinem, int J, std::string id = "elastic_scattering")
            : raw_partial_wave(key, xkinem, J, id)
            {
                initialize(2);
            };

            // -----------------------------------------------------------------------
            // REQUIRED VIRTUAL FUNCTIONS

            inline complex helicity_amplitude(std::array<int,4> helicities, double s, double t)
            {
                store(helicities, s, t);
                return (2*_J+1) * legendre(_J, _kinematics->z_s(s,t)) * partial_wave(_s);
            };

            inline helicity_frame native_helicity_frame(){ return HELICITY_INDEPENDENT; };

            // Specify which final state particles amplitude can acommodate
            inline std::vector<particle> allowed_mesons() { return any(); };
            inline std::vector<particle> allowed_baryons(){ return any(); };

            inline void allocate_parameters(std::vector<double> pars)
            {
                _a = pars[0];
                if (_option == EffectiveRange) _b = pars[1];
            };

            inline complex partial_wave(double s)
            {
                // Calculate momentum dependence
                _q2 = q2(); _G = G();

                // K-matrix
                _K  = pow(_q2, _J)*(_a + _b*_q2);

                // T-amplitude
                _T  = _K/(1+_G*_K);
                return _T;
            };

            // -----------------------------------------------------------------------
            // OPTIONAL VIRTUAL FUNCTIONS

            // Assign each parameter a name, useful for fitting utlities
            inline std::vector<std::string> parameter_labels()
            {
                std::vector<std::string> labels = {"a"};
                if (_option == EffectiveRange) labels.push_back("b");
                return labels;
            };

            inline void set_option(int x)
            {
                switch (x)
                {
                    case kScatteringLength:
                    {
                        set_N_pars(1);
                        _b = 0; break;
                    };
                    case kEffectiveRange:   
                    { 
                        set_N_pars(2); break;
                    };
                    default: option_error(); return;
                };

                _option = x;
            }

            // Options
            static const int kScatteringLength = 0;
            static const int kEffectiveRange   = 1;

            protected:
            
            // Terms in the K-matrix
            double _a = 0, _b = 0;

            // Intermediate variables
            double _q2 = 0;
            complex _K = 0, _G = 0, _T = 0;

            // Chew-Mandelstam functions
            inline complex G()
            {
                double m1 = _mX, m2 = _mR;
                complex rho, xi;
                complex result;

                rho    = csqrt(Kallen(_s, m1*m1, m2*m2)) / _s;
                xi     = 1 - (m1+m2)*(m1+m2)/_s;
                result = (rho*log((xi + rho) / (xi - rho)) - xi*(m2-m1)/(m2+m1)*log(m2/m1)) / PI;
                return result / (16*PI);
            };

            // Momentum squared
            inline double q2()
            {
                double m1 = _mX, m2 = _mR;
                return Kallen(_s, m1*m1, m2*m2) / (4*_s);
            };
        };
    };
};

#endif