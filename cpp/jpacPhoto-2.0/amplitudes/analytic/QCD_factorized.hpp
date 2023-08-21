// Implementation of vector meson photoproduction from QCD factorizaton in [1]
// The amplitude is related to moments of the GPD function near threshold
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------
// References: 
// [1] arXiv:2103.11506 [hep-ph]
// ------------------------------------------------------------------------------


#ifndef ANALYTIC_QCDFACTORIZED_HPP
#define ANALYTIC_QCDFACTORIZED_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "amplitude.hpp"
#include "cgamma.hpp"

namespace jpacPhoto
{
    namespace analytic
    {
        class QCD_factorized : public raw_amplitude
        {
            public: 

            QCD_factorized(amplitude_key key, kinematics xkinem, std::string id = "QCD_factorized")
            : raw_amplitude(key, xkinem, id)
            {
                initialize(4);
            };

            // -----------------------------------------------------------------------
            // Virtuals 

            // We assume spinless particles so we have no helicity structure
            inline complex helicity_amplitude(std::array<int,4> helicities, double s, double t)
            {
                // Save inputes
                store(helicities, s, t);

                _xi = xi();

                // Calculate form factors at fixed t 
                _Ag = _Ag0 / pow( 1 - _t/_mA/_mA, 2);
                _Bg = 0;
                _Cg = _Cg0 / pow( 1 - _t/_mC/_mC, 2);

                // GPD scalar functions
                _H2 = _Ag + 4*_xi*_xi*_Cg;
                _E2 = _Bg - 4*_xi*_xi*_Cg;

                // Calculate the spin-summed amplitude
                double Gsq = ((1 -_t/(_mT + _mR))*_E2*_E2 - 2*_E2*(_H2 + _E2) + (1-_xi*_xi)*(_H2 + _E2)*(_H2 + _E2))/ pow(_xi, 4);

                // Return square root 
                return _phi0 * sqrt(Gsq) /* * _norm / pow(_mX, 3/2) */;
            };

            // Even though its "analytic" we require s-channel helicity conservation
            inline helicity_frame native_helicity_frame(){ return HELICITY_INDEPENDENT; };

            // Vector mesons and half plus only
            inline std::vector<particle> allowed_mesons() { return { VECTOR }; };
            inline std::vector<particle> allowed_baryons(){ return { HALFPLUS }; };

            // Parameter names are a[J] and b[J] for scattering length and normalization respectively
            inline std::vector<std::string> parameter_labels()
            {
                return { "|phi(0)|", "mA", "Ag(0)", "mC", "Cg(0)" }; 
            };
           
            protected:

            inline void allocate_parameters(std::vector<double> pars)
            {
                _phi0 = pars[0];
                _mA   = pars[1];
                _Ag0  = pars[2];
                _mC   = pars[3];
                _Cg0  = pars[4];
            };

            // Light-cone variables
            inline double P_plus_i()
            {
                return (_kinematics->target_energy(_s) + _kinematics->initial_momentum(_s)) / sqrt(2);
            };

            inline double P_plus_f()
            {
                return (_kinematics->recoil_energy(_s) + _kinematics->final_momentum(_s) * cos(_theta)) / sqrt(2);
            };

            // Skewness
            inline double xi()
            {
                double pi = P_plus_i(), pf = P_plus_f();
                return (pi - pf) / (pi + pf);
            };

            // Free parameters
            double _phi0 = 1.0952/(4*PI);      // Quarkonium wave function at threshold
            double _mA = 1.64, _mC = 0.48;     // Dipole masses
            double _Ag0 = 0.58, _Cg0 = -0.84;  // Dipole normalizations

            // Intermediate variables
            double _xi = 0;                    // Skewness
            double _Ag = 0, _Bg = 0, _Cg = 0;  // Gravitational form factors
            double _H2 = 0, _E2 = 0;           // Scalar GPD moments

            // Normalization factors
            double _alpha_EM = 1/137, _alpha_S = 0.3;
            double _e = sqrt(4*PI*_alpha_EM);
            double _norm = 2/3*_e*_e*(16*PI*_alpha_S)/sqrt(3);
        };
    };
};

#endif