// Exchange diagrams from effective lagrangians often utilize 'form-factors'
// to parameterize ignorance of off-shell propagators
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef FORM_FACTOR_HPP
#define FORM_FACTOR_HPP

#include <memory>
#include "constants.hpp"

namespace jpacPhoto
{

    // We only ever want these to be saved as internal objects of an amplitude
    // so make them unique_ptrs so they may be swapped out as options change 
    // but also garbage collected when the amplitude goes out of scope
    class raw_form_factor;
    using form_factor = std::unique_ptr<raw_form_factor>;

    template<class F>
    inline form_factor new_FF()
    {
        form_factor ptr = std::make_unique<F>();
        return ptr;
    };

    template<class F>
    inline form_factor new_FF(double onshell_mass)
    {
        form_factor ptr = std::make_unique<F>(onshell_mass);
        return ptr;
    };

    // Calling this by itself will bring up the default "no FF" form_factor
    inline form_factor new_FF()
    {
        return std::make_unique<raw_form_factor>();
    };

    // Abstract class which all 
    class raw_form_factor 
    {
        public:
        
        raw_form_factor()
        {};

        raw_form_factor(double onshell_mass)
        : _onshell(onshell_mass)
        {};

        inline void set_cutoff(double cutoff)
        {
            _cutoff = cutoff;
        };

        virtual complex eval(double s){ return 1; };

        protected:

        double _onshell = 0;
        double _cutoff  = 0;
    };  

    // ------------------------------------------------------------------------------
    // Common parameterizations of the form factor include :

    // Exponential form-factor common in Regge exchange physics at high-energies
    // Does not care about the on-shell mass of the particle as we are expected to be 
    // at high energies far from the pole
    struct exponential : public raw_form_factor
    {
        exponential()
        : raw_form_factor()
        {};

        inline complex eval(double t){ return exp( XR * t / (_cutoff*_cutoff) ); };
    };

    // Monopole form-factor which mimics the power-law behavior at small momentum transfers
    // This modifies the propagator with additional powers which vanish at the on-shell point
    struct monopole : public raw_form_factor
    {
        monopole(double onshell)
        : raw_form_factor(onshell)
        {};

        inline complex eval(double t){ return (_cutoff*_cutoff - _onshell*_onshell) / (_cutoff*_cutoff - t); };
    };

};

#endif