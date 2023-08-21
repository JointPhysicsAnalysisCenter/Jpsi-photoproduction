// This is a template file to illustrate basic requirements to define a new amplitude 
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef TEMPLATE_HPP
#define TEMPLATE_HPP

// Basic includes
#include "constants.hpp"
#include "kinematics.hpp"
#include "amplitude.hpp"

// Amplitudes should be added to the jpacPhoto namespace
namespace jpacPhoto
{
    // Additionally a second namespace may be used to differentiate different
    // implementations of the same amplitude
    // e.g. covariant::my_amplitude and analytic::my_amplitude

    // All amplitude implementations must derive from raw_amplitude
    // Every amplitude is carries a string id in the constructor to identify it from other amplitudes when plotting/fitting/error handling
    class my_amplitude : public raw_amplitude
    {
        public: 

        // Basic constructor
        my_amplitude(amplitude_key key, kinematics xkinem, std::string id = "my_amplitude's default id")
        : raw_amplitude(key, xkinem, id)
        {
            // Constructor should initialize with number of parameters
            // this specifies 2 free parameters
            initialize(2);

            // Do anything else you want at construction
        };

        // Additional constructors can be used with up to three additional parameters of arbitrary type
        // Thes should always come after kinematics but before the id which always comes last
        my_amplitude(amplitude_key key, kinematics xkinem, sometype local_data, std::string id = "my_amplitude's default id")
        : raw_amplitude(key, xkinem, id)
        {
            initialize(2);

            // Do something with local_data
        };

        my_amplitude(amplitude_key key, kinematics xkinem, sometype data1, othertype data2, std::string id = "my_amplitude's default id")
        : raw_amplitude(key, xkinem, id)
        {
            initialize(2);

            // Do something 
        };

        // -----------------------------------------------------------------------
        // REQUIRED VIRTUAL FUNCTIONS

        // Provide a way to compute the amplitude at fixed helicities, s, and t
        inline complex helicity_amplitude(std::array<int,4> helicities, double s, double t)
        {
            // jpacPhoto::raw_amplitude has protected member data
            // _helicities, _s, and _t
            // which can be accessed from other functions without needed to be passed as argument
            // the store function saves them;
            store(helicities, s, t);

            complex result;
            /* compute amplitude */

            return result;
        };

        // Must specifiy which center-of-mass scattering channel the helicities are defined
        // This is required to correctly give parity-relations and polarization observables
        inline helicity_channel native_helicity_frame(){ return helicity_channel::S_CHANNEL; };

        // Specify which final state particles amplitude can acommodate
        inline std::vector<particle> allowed_mesons() { return { vector, axialvector }; };
        inline std::vector<particle> allowed_baryons(){ return { halfplus, halfminus }; };

        // Given a std::vector<double> of size N_pars
        // save parameters to local variables with which to calculate amplitude
        // The parameters are set by user script with raw_amplitude::set_parameters() which 
        // will do a check the input is of appropriate size and then call 
        // raw_amplitude::allocate_parameters internally
        inline void allocate_parameters(std::vector<double> pars)
        {
            /* save pars somewhere */
            _p1 = pars[0];
            _p2 = pars[1];
        };

        // -----------------------------------------------------------------------
        // OPTIONAL VIRTUAL FUNCTIONS

        // Assign each parameter a name, useful for fitting utlities
        inline std::vector<std::string> parameter_labels()
        {
            return {"p1", "p2" /*, etc...*/ };
        };
        
        // Use amplitude_option's to turn on and off features of your amplitude
        inline void set_option(amplitude_option x)
        {
            /* e.g. change shape of a form factor */
            if (x == ExponentialFF){ /* do something*/;}
            else if (x == MonopoleFF){/* do something else */;}
        };

        // -----------------------------------------------------------------------
        // Other than the required virtual functions structure can be whatever you want

        inline void new_function()
        {
            /* do something not incorporated in the parent raw_amplitude class */
        };

        // Note that amplitudes can only be created through the make_amplitude<T> method
        // This means that public non-virtual functions arent accessable from an amplitude object
        // To access class-specific member data you need to downcast, e.g.:

        // amplitude amp = make_amplitude<my_amplitude>(kinem);
        // auto downcasted_amp = std::dynamic_pointer_cast<K_matrix>(amp);
        // downcasted_amp->new_function();

        protected:

        // Local data
        double _p1 = 0, _p2 = 0;
    };

#endif