
// Implementation of a simple pomeron exchange including vector-pomeron like covariant structure
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef COVARIANT_POMERON_HPP
#define COVARIANT_POMERON_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "amplitude.hpp"

namespace jpacPhoto
{
    namespace covariant
    {
        class pomeron_exchange : public analytic::pomeron_exchange
        {
            public: 

            pomeron_exchange(amplitude_key key, kinematics xkinem, std::string id = "pomeron_exchange")
            : analytic::pomeron_exchange(key, xkinem, id)
            {};

            // -----------------------------------------------------------------------
            // Virtuals 

            // Everything is the same as the analytic::pomeron model with the exception
            // of replacing the helicty conserving delta_function with the covariant structure
            inline complex helicity_amplitude(std::array<int,4> helicities, double s, double t)
            {
                // Save inputes
                store(helicities, s, t);
                _covariants->update(helicities, s, t);

                // Intermediate quantities
                double t_prime = t - _kinematics->t_min(s);
                double alpha   = _alpha0 + t * _alphaP;

                // Helicity structure comes from contractin the top and bottom vertices
                // We remove one factor of s to account for the asymptotic scaling of the dirac_spinors
                complex helicity_structure = contract(top_vertex(), bottom_vertex()) / _s;

                return _A * exp(_b0 * t_prime) * pow((s - _kinematics->sth()) / _s0, alpha) * helicity_structure;
            };

            // The covariant structure explicitly defined in s-channel
            inline helicity_frame native_helicity_frame(){ return S_CHANNEL; };

            // Vector mesons and half plus only
            inline std::vector<particle> allowed_mesons() { return { VECTOR }; };
            inline std::vector<particle> allowed_baryons(){ return { HALFPLUS }; };

            // -----------------------------------------------------------------------
            protected:

            // Photon -- Pomeron -- Vector meson vertex
            inline lorentz_tensor<complex,1> top_vertex()
            {
                // Beam polarization and momentum
                auto eps   = _covariants->eps();
                auto q     = _covariants->q();

                // Vector polarization
                auto eps_p = _covariants->eps_prime();

                return contract(eps, eps_p) * q - contract(q, eps_p) * eps;
            };  

            // Nucleon -- Pomeron -- Nucleon vertex
            inline lorentz_tensor<complex,1> bottom_vertex()
            {
                // Spinors
                auto u    = _covariants->u();    // Target
                auto ubar = _covariants->ubar(); // Recoil

                return bilinear(ubar, gamma_vector(), u);;
            };
        };
    };
};

#endif