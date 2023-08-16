// Container class for covariant quantities!
// Since not all amplitudes need these they're seperated for convience
//
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@iu.edu
// ---------------------------------------------------------------------------

#include "covariants.hpp"
#include "dirac_spinor.hpp"

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // Methods related to our caching to minimize calculations

    // Check if energy step requires recalculating quantities
    void covariants::check_cache(double s, double t)
    {
        bool s_changed      =   unsynced(s, _s);
        bool masses_changed =   unsynced(_kinematics->get_beam_mass(), _mB)
                             || unsynced(_kinematics->get_target_mass(), _mT)
                             || unsynced(_kinematics->get_meson_mass(), _mX)
                             || unsynced(_kinematics->get_recoil_mass(), _mR);

        if ( s_changed || masses_changed ) recalculate(s);

        // Recalculating theta dependence is super easy 
        // so this doesnt need a seperate method
        double theta = _kinematics->theta_s(s, t);
        if ( unsynced(theta, _theta) )
        {
            _theta = theta;
            _cos = cos(theta);
            _sin = sin(theta);
            _coshalf = cos(theta/2);
            _sinhalf = sin(theta/2);
        };

        return;
    };

    // The complicated recalculation only comes with changing energy dependence
    void covariants::recalculate(double s)
    {
        // save the latest masses
        _mB = _kinematics->get_beam_mass();
        _mT = _kinematics->get_target_mass();
        _mX = _kinematics->get_meson_mass();
        _mR = _kinematics->get_recoil_mass();
        _s  = s;

        // then update momenta
        _qi = _kinematics->initial_momentum(s);
        _qf = _kinematics->final_momentum(s);

        // and energies
        _EB = _kinematics->beam_energy(s);
        _ET = _kinematics->target_energy(s);
        _EX = _kinematics->meson_energy(s);
        _ER = _kinematics->recoil_energy(s);
    };

    // ---------------------------------------------------------------------------
    // FOUR MOMENTA 

    // Beam momentum
    // Aligned with the positive z-hat axis
    lorentz_tensor<complex,1> covariants::q()
    {   
        return lorentz_vector<complex>( {{_EB, 0., 0., _qi}} );
    };

    // Target momentum
    // 3-momenta opposite q in the -z-hat direction
    lorentz_tensor<complex,1> covariants::p()
    {   
        return lorentz_vector<complex>( {{_ET, 0., 0., -_qi}} );
    };

    // Produced meson momentum
    // In x-z plane with axis angle _theta from the +z
    lorentz_tensor<complex,1> covariants::q_prime()
    {
        return lorentz_vector<complex>( {{_EX, _qf*_sin, 0., _qf*_cos}} );
    };

    // Recoil baryon momentum
    // In x-z plane with axis angle _theta from the -z
    lorentz_tensor<complex,1> covariants::p_prime()
    {
        return lorentz_vector<complex>( {{_ER, -_qf*_sin, 0., -_qf*_cos}} );
    };

    // ---------------------------------------------------------------------------
    // POLARIZATION VECTORS
    
    // Incoming, beam polarization vector
    lorentz_tensor<complex,1> covariants::eps()
    {
        // If helicity out of bounds return zero vector
        if (abs(_lamB) > 1) return error("eps", "Invalid helicity passed!", lorentz_vector<complex>({{0,0,0,0}}));

        // Check cases:
        bool transverse = ( abs(_lamB) == 1 );
        bool massive    = !_kinematics->is_photon();
        
        if ( transverse ) return - lorentz_vector<complex>({{  0, _lamB,  I,   0}}) / sqrt(2);
        if ( massive )    return   lorentz_vector<complex>({{_qi,     0,  0, _EB}}) / _mB;
        else              return   lorentz_vector<complex>({{  0,     0,  0,   0}});
    };

    // Outgoing, meson polarization vector
    lorentz_tensor<complex,1> covariants::eps_prime()
    {
        // If helicity out of bounds return zero vector
        if (abs(_lamX) > 1) return error("eps_prime", "Invalid helicity passed!", lorentz_vector<complex>({{0,0,0,0}}));

        // Currently doubly massless particles (compton scattering) is not available 
        if ( is_zero(_mX) ) return error("eps_prime", "Massless final state boson not supported yet!", lorentz_vector<complex>({{0,0,0,0}}));
        
        // Check cases:
        bool transverse = ( abs(_lamX) == 1 );
            
        if ( transverse ) return - lorentz_vector<complex>({{  0, _lamX*_cos,  I, -_lamX*_sin}}) / sqrt(2);
        else              return   lorentz_vector<complex>({{_qf,   _EX*_sin,  0,    _EX*_cos}}) / _mX;
    };

    // ---------------------------------------------------------------------------
    // DIRAC SPINORS

    // incoming (target) spinor
    dirac_spinor covariants::u()
    {
        complex wp = csqrt(_ET + _mT), wm = csqrt(_ET - _mT);

        if (_lamT == 1) return - dirac_spinor({{{0, wp,  0, wm}}});
        else            return   dirac_spinor({{{wp, 0, -wm, 0}}});
    };

    // outgoing (recoil) spinor
    dirac_spinor covariants::ubar()
    {
        complex wp = csqrt(_ER + _mR), wm = csqrt(_ER - _mR);
        
        dirac_spinor u = (_lamR == 1) ? dirac_spinor({{{wp*_sinhalf, -wp*_coshalf,  wm*_sinhalf, -wm*_coshalf}}}) 
                                      : dirac_spinor({{{wp*_coshalf,  wp*_sinhalf, -wm*_coshalf, -wm*_sinhalf}}});

        return u.adjoint();
    };
};
