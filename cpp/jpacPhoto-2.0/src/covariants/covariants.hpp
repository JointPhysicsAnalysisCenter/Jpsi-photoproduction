// Container class for covariant quantities!
// Since not all amplitudes need these they're seperated for convience
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@iu.edu
// ------------------------------------------------------------------------------

#ifndef COVARIANTS_HPP
#define COVARIANTS_HPP

#include "constants.hpp"
#include "kinematics.hpp"
#include "lorentz_tensor.hpp"
#include "dirac_spinor.hpp"
#include "contract.hpp"
#include "bilinear.hpp"

namespace jpacPhoto
{   
    // Now the covariants class assembles all relavant kinematic quantities and outputs 
    // them in the form they can be assembled in to amplitudes with Feynman rules
    class covariants
    {
        // ---------------------------------------------------------------------------
        public: 

        covariants(kinematics xkinem)
        : _kinematics(xkinem)
        {};

        // Just like the amplitudes here we can save s, t, and helicities to avoid carrying them around
        inline void update(std::array<int,4> helicities, double s, double t)
        {
            // Save the helicities to avoid having to carry them around
            _lamB = helicities[0];
            _lamT = helicities[1];
            _lamX = helicities[2];
            _lamR = helicities[3];

            // Recalculate our cache if things have changed
            check_cache(s, t);            
        };  

        // Same as above but dont update the helicities
        inline void update(double s, double t)
        {
            // Recalculate our cache if things have changed
            check_cache(s, t);            
        };  

        // ---------------------------------------------------------------------------
        // These are the primary methods. They use the cached information and output
        // the appropriate Lorentz and Dirac structures

        // FOUR MOMENTA 
        // q Always refers to mesons, p always baryons
        // prime indicates final state

        lorentz_tensor<complex,1> q();         // Beam momentum 
        lorentz_tensor<complex,1> q_prime();   // Meson momentum 
        lorentz_tensor<complex,1> p();         // Target momentum 
        lorentz_tensor<complex,1> p_prime();   // Recoil momentum 

        // Momentum transfers  
        inline lorentz_tensor<complex,1> k_s(){ return q() + p(); };        // s-channel momentum transfer
        inline lorentz_tensor<complex,1> k_t(){ return q() - q_prime(); };  // t-channel momentum transfer
        inline lorentz_tensor<complex,1> k_u(){ return q() - p_prime(); };  // u-channel momentum transfer

        // Polarization vectors
        lorentz_tensor<complex,1> eps();       // Beam (incoming) polarization
        lorentz_tensor<complex,1> eps_prime(); // Meson (outgoing) polarization

        // Field strength tensor
        inline lorentz_tensor<complex,2> F(){ return tensor_product(q(), eps()) - tensor_product(eps(), q()); }; // Beam field tensor

        // SPINORS
        dirac_spinor u();    // incoming (target) spinor
        dirac_spinor ubar(); // outgoing (recoil) spinor

        // ---------------------------------------------------------------------------
        private:

        // Save helicities current helicities so they dont need to be passed around
        int _lamB, _lamT, _lamX, _lamR; 

        // External helicities saved internally so they dont need to be specified as arguments
        kinematics _kinematics;

        void check_cache(double s, double t);
        void recalculate(double s);

        // Caching tolerance
        constexpr static double _tolerance = 1.E-5;
        inline bool unsynced(double x, double y)
        { return ( std::abs(x - y) < _tolerance ) ? false : true; };

        // Primary energy and angular variables
        double _s, _theta;

        // Masses 
        double _mB, _mT, _mX, _mR;

        // Energy of all three particles
        complex _EB, _ET, _EX, _ER;

        // 3-momenta of initial and final states
        complex _qi, _qf;

        // cos(theta) and sin(theta) of s-channel scattering angle
        double _cos, _sin;
        double _coshalf, _sinhalf;
    };
};

#endif