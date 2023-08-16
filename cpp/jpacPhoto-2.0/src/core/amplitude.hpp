// The core object of interest is the amplitude class which encorporates
// the dynamical physics model for a process
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef AMPLITUDE_HPP
#define AMPLITUDE_HPP

#include <memory>
#include <string>

#include "constants.hpp"
#include "helicities.hpp"
#include "kinematics.hpp"
#include "covariants.hpp"
#include "angular_functions.hpp"

#include "Math/GSLIntegrator.h"
#include "Math/IntegrationTypes.h"
#include "Math/Functor.h"

namespace jpacPhoto
{   
    // Forward declare amplitude for the typedef below
    class raw_amplitude;

    // We only ever want to work with amplitudes as pointers on the heap.
    // This will allow our amplitudes to be added / passes around without issue
    using amplitude  = std::shared_ptr<raw_amplitude>;

    // We require this private class to allow only the new_amplitude function from
    // Creating amplitude objects
    class amplitude_key
    {
        private: 
        amplitude_key()
        {};

        // The only functions that can create new amplitudes
        // are these friend functions

        // Constructor with no free parameters
        template<class A>
        friend amplitude new_amplitude(kinematics);

        // non-trivial id
        template<class A>
        friend amplitude new_amplitude(kinematics, std::string);

        // One additional abitrary parameter
        template<class A, typename B>
        friend amplitude new_amplitude(kinematics, B, std::string);

        // Two additional parameters
        template<class A, typename B, typename C>
        friend amplitude new_amplitude(kinematics, B, C, std::string);

        // Add two amplitude together
        friend amplitude operator+(amplitude a, amplitude b);

        // Project an amplitude onto partial wave with integer or half-integer spin
        friend amplitude project(int, amplitude, std::string);
        friend amplitude helicity_project(int, amplitude, std::string);
    };

    // This function serves as our "constructor"
    template<class A>
    inline amplitude new_amplitude(kinematics xkinem)
    {
        auto amp = std::make_shared<A>(amplitude_key(), xkinem);
        return std::static_pointer_cast<raw_amplitude>(amp);
    };

    template<class A>
    inline amplitude new_amplitude(kinematics xkinem, std::string id)
    {
        auto amp = std::make_shared<A>(amplitude_key(), xkinem, id);
        return std::static_pointer_cast<raw_amplitude>(amp);
    };

    // "constructor" specifying an extra parameter
    template<class A, typename B>
    inline amplitude new_amplitude(kinematics xkinem, B extra, std::string id)
    {
        auto amp = std::make_shared<A>(amplitude_key(), xkinem, extra, id);
        return std::static_pointer_cast<raw_amplitude>(amp);
    };

    // "constructor" specifying two extra parameters
    template<class A, typename B, typename C>
    inline amplitude new_amplitude(kinematics xkinem, B extra1, C extra2, std::string id)
    {
        auto amp = std::make_shared<A>(amplitude_key(), xkinem, extra1, extra2, id);
        return std::static_pointer_cast<raw_amplitude>(amp);
    };

    // ---------------------------------------------------------------------------
    // Opreations to sum amplitudes together

    // Check if two amplitudes are compatible
    bool are_compatible(amplitude a, amplitude b);

    // "Constructor" for a sum of amplituders
    amplitude operator+(amplitude a, amplitude b);

    // Shortcut to add to an existing sum
    void operator+=(amplitude a, amplitude b);

    // ---------------------------------------------------------------------------
    // The underlying class, which defines all our observables, etc

    class raw_amplitude
    {
        public:
        
        // Public constructor as required to use make_shared
        // but requires the amplitude_key as a parameter which is private
        // So this constructor is actually not accessable other than through
        // new_amplitude
        raw_amplitude(amplitude_key key, kinematics xkinem, std::string id = "amplitude")
        : _kinematics(xkinem), _id(id), _covariants(std::make_unique<covariants>(xkinem))
        {};        

        // Sum constructor 
        raw_amplitude(amplitude_key key, std::vector<amplitude> subamps, std::string id)
        : _id(id), _covariants(nullptr)
        {
            add(subamps);
        };

        // Access to the kinematics object we have stored
        kinematics _kinematics = nullptr;

        // ---------------------------------------------------------------------------
        // Virtual functions which must be defined by a given model

        // Given set of helicities, com invariant mass, and momentum transfer calculate the
        // given helicitiy amplitude
        // By default we assume theres is a sum of amplitudes where the total helicity amplitude
        // is the sum of individual amplitudes at fixed helicities
        virtual complex helicity_amplitude(std::array<int,4> helicities, double s, double t)
        {
            if (_subamplitudes.size() == 0)
            {
                return 0;
            }

            complex sum = 0;
            for (amplitude amp : _subamplitudes)
            {
                sum += amp->helicity_amplitude(helicities, s, t);
            }

            return sum;
        };
        
        // Each amplitude must be able to specify what channel it expects its helicities to be
        // defined with respect to (s, t, or u channel CoM frame)
        // By default all subamplitudes are assumed to have the same native frame (else it doesnt make sense
        // to sum them) thus we return the first in the vector.
        virtual helicity_frame native_helicity_frame()
        {
            return _subamplitudes[0]->native_helicity_frame();
        };

        // Additionally, given a kinematics with arbitrary final state quantum numbers
        // Amplitudes should be able to specify which spin-parity combinations 
        // they can accomodate
        virtual std::vector<particle> allowed_mesons(){  return _subamplitudes[0]->allowed_mesons(); };
        virtual std::vector<particle> allowed_baryons(){ return _subamplitudes[0]->allowed_baryons(); };

        // If an an appropriate flag is passed, make appropriate changes.
        // By default this does nothing except save _option
        virtual void set_option(int x){ _option = x; };
        
        // Give each parameter a name if you'd like
        virtual std::vector<std::string> parameter_labels()
        {
            std::vector<std::string> labels;
            if (_subamplitudes.size() == 0)
            {
                for (int i = 0; i < _N_pars; i++)
                {
                    labels.push_back("par[" + std::to_string(i) + "]");
                }
            }
            else
            {
                for (auto amp : _subamplitudes)
                {
                    std::vector<std::string> sublabels = amp->parameter_labels();
                    labels.insert(
                        labels.end(), sublabels.begin(), sublabels.end()
                    );
                };
            }

            return labels;
        };

        // ---------------------------------------------------------------------------
        // Initialization function to do checks and set option to default

        inline void initialize(int Npars = 0)
        {
            set_N_pars(Npars);
            check_QNs(_kinematics);
            set_option(0);
        };

        // ---------------------------------------------------------------------------
        // Crossing properties of helicity amplitudes can be accessed here

        // Phase relating the negation of helicities through parity symmetry
        double parity_phase(std::array<int,4> helicities);
        double parity_phase(int i);

        // ---------------------------------------------------------------------------
        // Observables
        // Evaluatable in terms of s and t (not angle!)

        // Modulus of the amplitude summed over all helicity combinations
        double probability_distribution(double s, double t);

        // Differential and total cross-section
        double differential_xsection(double s, double t);

        // integrated crossection
        double integrated_xsection(double s);

        // Spin asymmetries
        double A_LL(double s, double t); // Beam and target
        double K_LL(double s, double t); // Beam and recoil

        // Spin density matrix elements

        // this calculates the SDME in the amplitudes 'natural' frame 
        // i.e. doesnt require any additional rotations
        complex SDME(int alpha, int lam, int lamp, double s, double t);

        // Rotate the SDMEs to another frame by and angle theta
        complex rotated_SDME(int alpha, int lam, int lamp, double s, double t, double theta);
       
        // These check what the natural frame and include rotations if required to ensure
        // SDME is defined in the Helicity or Gottfried-Jackson frames
        complex SDME_H (int alpha, int lam, int lamp, double s, double t);
        complex SDME_GJ(int alpha, int lam, int lamp, double s, double t);

        // Beam Asymmetries
        double beam_asymmetry_y(double s, double t);    // Along the y direction
        double beam_asymmetry_4pi(double s, double t);  // integrated over decay angles

        // Parity asymmetry
        double parity_asymmetry(double s, double t);

        // ---------------------------------------------------------------------------
        // Other miscellaneous getters and setters

        // Access or change the id 
        inline void set_id(std::string id){ _id = id; };
        inline std::string id(){ return _id; };

        // Change the debug flag
        inline void set_debug(int x)
        { 
            _debug = x;
            for (auto amp : _subamplitudes)
            {
                amp->set_debug(x);
            }
        };

        // Number of free parameters
        inline int N_pars(){ return _N_pars; };

        // This function is what a user actually calls
        // It wraps the protected vitual method allocate_parameters() with checks of correct size and caching
        inline void set_parameters( std::vector<double> x )
        {
            // Check the size is alright
            if (!correct_size(x))
            {
                pars_error(x.size());
                return;
            };

            // Allocate them (amplitude specific)
            allocate_parameters(x);

            // Notify our cache that changes have been made
            _parameters_changed = true;
        };

        // Given a vector of double of appropriate length, allocate free parameters to model
        // By default we feed the total parameters into the subamplitudes 
        virtual void allocate_parameters( std::vector<double> x )
        {
            // For each subamplitude grab a subvector of its parameters
            auto running_iter = x.begin();
            for (amplitude amp : _subamplitudes)
            {
                auto sub_pars = std::vector<double>(running_iter, running_iter + amp->N_pars());
                amp->set_parameters(sub_pars);

                running_iter += amp->N_pars();
            };
        };

        // ---------------------------------------------------------------------------
        protected:
        
        // String identifier
        std::string _id = "amplitude";

        // Debugging flag
        unsigned int _debug = 0;

        // Each amplitude may have different options for evaluating their amplitude
        // they may be differenticated with this variable
        int _option = 0;
        inline void option_error()
        {
            warning(id()+"::set_option", 
                    "Unexpected option passed. Continuing without change..."); 
        };

        // ---------------------------------------------------------------------------
        // Parameter handling 

        // Number of parameters to expect
        int  _N_pars = 0;

        // Whether parameters have changed since the last caching
        bool _parameters_changed = false;

        // Ability to change number of expected parameters from inside a class
        void set_N_pars(int N);

        // Simple check that a given vector is of the expected size
        bool correct_size(std::vector<double> pars);
        inline void pars_error(int x)
        {
            warning(id()+"::set_parameters", 
                    "Unexpected number of parameters passed. Expected "+std::to_string(_N_pars)+" but recieved "+std::to_string(x)+").");
        };

        // ---------------------------------------------------------------------------
        // Check different compatibilites

        // At construction, check the kinematics with set quantum numbers
        // is compatable with the amplitude
        void check_QNs(kinematics kinem);

        // ---------------------------------------------------------------------------
        // Each amplitude may also store copies of all kinematic variables
        // to avoid having to pass them around

        void store(std::array<int,4> helicities, double s, double t);

        // Invariant masses and scattering angle
        double _s, _t, _theta;

        // External particle masses
        double _mB, _mX, _mR, _mT;

        // Helicities
        int _lamB, _lamX, _lamR, _lamT;

        // Store of covariant quantities
        std::unique_ptr<covariants> _covariants;

        // ---------------------------------------------------------------------------
        // When calculating observables we can cache helicity amplitudes to avoid
        // unnecessary calculation

        // Update the cache
        void update_cache(double s, double t);

        double _cache_tolerance = 1.E-4;
        double _cached_mX = 0., _cached_mR = 0.; // Final state masses 
        double _cached_s  = 0., _cached_t  = 0.; // Invariants
        
        // Saved copies of amplitudes
        std::vector<complex> _cached_helicity_amplitudes;

        // ---------------------------------------------------------------------------
        // Methods and data related to summing amplitudes together
        // These are private because we dont want any derived classes to act like sums

        private:

        // If the current pointer is a sum or not
        inline bool is_sum() { return (_subamplitudes.size() > 0); };

        // If it is we call this vector of amplitudes to calculate helicity amplitudes
        std::vector<amplitude> _subamplitudes;

        // Check if a given amplitude is compatible to sum with *this
        bool is_compatible(amplitude new_amp);

        // add a new amplitude to the list in a sum
        void add(amplitude new_amp);

        // or add a whole vector worth of new amplitudes
        void add(std::vector<amplitude> new_amps);

        // Friend to allow this operator to acess add()
        friend void operator+=(amplitude a, amplitude b);
        friend amplitude operator+(amplitude a, amplitude b);
        friend std::vector<amplitude> extract_subamplitudes(amplitude amp);
    };

    // Given an amplitude, check if its a sum of amplitudes
    // if so, return vector of constituent subamplitudes
    // if not, return amplitude vector with only itself
    inline std::vector<amplitude> extract_subamplitudes(amplitude amp)
    {        
        if (!amp->is_sum()) return {amp};
        return amp->_subamplitudes;
    };

    // Extracts sub_amplitudes but addtionally appends the sum to the vector
    inline std::vector<amplitude> expand(amplitude amp)
    {
        std::vector<amplitude> expanded = extract_subamplitudes(amp);
        if (expanded.size() > 1) expanded.push_back(amp);
        return expanded;
    };
};

#endif