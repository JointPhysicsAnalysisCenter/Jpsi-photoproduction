// Extension of the raw_amplitude which supports individual partial waves in the s-channel
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef PARTIAL_WAVE_HPP
#define PARTIAL_WAVE_HPP

#include "constants.hpp"
#include "helicities.hpp"
#include "kinematics.hpp"
#include "amplitude.hpp"

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <memory>

namespace jpacPhoto
{
    // Foward declare the PW amplitude
    class raw_partial_wave;

    // Similar to amplitude we only ever want partial_waves to be pointers
    using partial_wave = std::shared_ptr<raw_partial_wave>;

    // Summing two partial waves defaults to a "full" amplitude
    amplitude operator+(partial_wave a, partial_wave b);

    // ---------------------------------------------------------------------------
    // Methods to make partial_Waves from existing amplitudes

    // "Constructor" function which projects an existing amplitude onto legendre polynomial
    amplitude project(int J, amplitude to_project, std::string id = "");

    // "Constructor" function which projects onto d-functions
    amplitude helicity_project(int J, amplitude to_project, std::string id = "");

    // ---------------------------------------------------------------------------
    // Raw_amplitude class

    class raw_partial_wave : public raw_amplitude
    {
        public:

        // This constructor should be used for any user defined derived classes
        raw_partial_wave(amplitude_key key, kinematics xkinem, int J, std::string id)
        : raw_amplitude(key, xkinem, id), 
          _J(J)
        {};

        // This constructor is specifically for use with the project() function
        raw_partial_wave(amplitude_key key, int J, amplitude to_project, bool if_halfint, std::string id)
        : raw_amplitude(key, to_project->_kinematics, id), 
          _J(J), _halfinteger(if_halfint), _amplitude(to_project)
        {
            set_N_pars(0);
        };

        // These are always assumed to be s-channel helicities so this is fixed
        helicity_frame native_helicity_frame()
        {
            return helicity_frame::S_CHANNEL;
        };

        virtual std::vector<particle> allowed_mesons(){  return (_amplitude == nullptr) ? std::vector<particle>() : _amplitude->allowed_mesons(); };
        virtual std::vector<particle> allowed_baryons(){ return (_amplitude == nullptr) ? std::vector<particle>() : _amplitude->allowed_baryons(); };
        
        // Return the J-th term to the full amplitude by multiplying by angular function
        // These may be overloaded with an explicit model for the PWA
        virtual complex helicity_amplitude(std::array<int,4> helicities, double s, double t);

        // By default we calculate the partial-wave projection integral numerically from 
        // the saved amplitude 
        virtual complex partial_wave(std::array<int,4> helicities, double s);
        virtual complex partial_wave(double s){ return partial_wave( {_lamB, _lamT, _lamX, _lamR}, _s); };

        
        // Output the J quantum number
        inline int J(){ return _J; };
        
        protected:

        // Produce a string of a parameter name which appends the J quantum number to it, i.e. "name[J]"
        inline std::string J_label(std::string name){ return name + "[" + std::to_string(_J) + "]"; };

        virtual inline void allocate_parameters(int x)
        {
            warning("partial_wave::allocate_parameters", 
                    "Partial-waves created using project() cannot change parameters! Change them in the amplitude being projected");
            return;
        };

        // By default partial_waves carry a pointer to a full amplitude and which they project
        amplitude _amplitude = nullptr;

        // Fixed spin identifier.
        // This may either be the whole-spin orbital angular momentum L
        // or half-integer total spin J
        int _J    = 0;
        bool _halfinteger = false;
    };
};

#endif