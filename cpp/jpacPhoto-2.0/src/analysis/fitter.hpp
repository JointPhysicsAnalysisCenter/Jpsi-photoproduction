// Class which allows an amplitude to be fit to data based on chi2 minimization
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#ifndef FITTER_HPP
#define FITTER_HPP

#include "kinematics.hpp"
#include "amplitude.hpp"
#include "data_set.hpp"

#include <chrono>
#include <string>
#include <vector>

#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "TRandom.h"

namespace jpacPhoto
{
    class fitter; 

    // ---------------------------------------------------------------------------
    // Structs for storing relevant info inside the fitter

    // Each free parameter of a model has associated with is a bunch of options
    class parameter
    {
        private:

        // Default initialized
        // This initializes a fixed parameter for normalizations
        parameter()
        : _is_norm(true),
          _fixed(true), _value(1.), 
          _label("norm"), _i(-1),
          _lower(0), _upper(5)
        {};
        
        // Constructor for amplitude variables
        parameter(int i)
        : _i(i), _label(default_label(i))
        {};

        int         _i;
        std::string _label;

        bool   _is_norm       = false;
        bool   _fixed         = false;
        double _value         = 0;
        bool   _custom_limits = false;
        double _upper         = 0;
        double _lower         = 0;
        double _step          = 0.1;

        friend class fitter;

        static inline std::string default_label(int i)
        {
            return "par[" + std::to_string(i) + "]";
        };
    };

    // ---------------------------------------------------------------------------
    // Actual fitter object

    class fitter
    {
        public:
        
        // Basic constructor, only requires amplitude to be fit 
        // uses default settings for minuit
        fitter(amplitude amp_to_fit)
        : _amplitude(amp_to_fit),
          _minuit(ROOT::Math::Factory::CreateMinimizer("Minuit2", "Combined"))
        {
            // Populate the _pars vector with the appropriate sized array
            reset_parameters();
        };

        // Parameterized constructor 
        // with explicit choice of minimization strategy and tolerance of minuit routines
        fitter(amplitude amp_to_fit, std::string strategy, double tolerance = 1.E-6)
        : _amplitude(amp_to_fit), _tolerance(tolerance),
          _minuit(ROOT::Math::Factory::CreateMinimizer("Minuit2", strategy))
        {
            reset_parameters();
        };

        // -----------------------------------------------------------------------
        // Methods to add data to be fit against

        // Parse a data set and add it to the fitting pool
        // Currently supported data_types are integrated and differential x-sections
        void add_data(data_set data);

        // OR pass in a whole vector and each one individually
        inline void add_data(std::vector<data_set> data)
        {
            for (auto set : data)
            {
                add_data(set);
            }
        };

        // Remove all saved data 
        void clear_data();

        // -----------------------------------------------------------------------
        // Set limits, labels, and fix parameters

        // Reset labels, limits and options on all parameters
        void reset_parameters();

        // Give each parameter a label beyond their default par[i] name
        void set_parameter_labels(std::vector<std::string> labels);

        // Set limits and/or a custom stepsize
        void set_parameter_limits(parameter& par, std::array<double,2> bounds, double step = 0.1);
        void set_parameter_limits(std::string label, std::array<double,2> bounds, double step = 0.1);

        // Indicate a parameter should be fixed to its initial guess value or a fixed val
        void fix_parameter(parameter& par, double val = 0);
        void fix_parameter(std::string label, double val = 0);

        // Unfix a parameter
        void free_parameter(parameter& par);
        void free_parameter(std::string label);

        // -----------------------------------------------------------------------
        // Methods related to fit options

        // Set the maximum number of calls minuit will do
        inline void set_max_calls(int n){ _max_calls = n; };
        
        // Message level for minuit (0-4)
        inline void set_print_level(int n){ _print_level = n; };

        // Change tolerance
        inline void set_tolerance(double tol){ _tolerance = tol; };

        // Change the guess range for initializing parameters
        inline void set_guess_range(std::array<double,2> bounds){ _guess_range = bounds; };

        // Actually do the fit given a vector of size amp->N_pars() as starting values
        // Prints results to command line but also returns the best-fit chi2 value
        void do_fit(std::vector<double> starting_guess, bool show_data = true);

        // Repeat do_fit N times and find the best fit
        // Parameters are randomly initialized each time on the interval [-5, 5] unless custom limits are set
        void do_fit(int N);

        // Return a vector of best-fit parameters from last fit
        inline std::vector<double> best_fit(){ return (_fit) ? _fit_pars : std::vector<double>(); };

        // Return a vector of best-fit parameters from last fit
        inline double chi2()   { return (_fit) ? _chi2    : -1; };
        inline double chi2dof(){ return (_fit) ? _chi2dof : -1; };

        // Return the best fit normalization
        inline double norm()   { return (_fit && !_norm._fixed) ? _best_norm : 1; }

        // Return the pointer to the amp being fit
        inline amplitude fit_amplitude(){ return _amplitude; };

        private:

        // This ptr should point to the amplitude to be fit
        amplitude _amplitude = nullptr;

        // -----------------------------------------------------------------------
        // Data handling

        // Total number of data points
        int _N = 0;

        std::vector<data_set> _int_data;  // Integrated x-section data
        std::vector<data_set> _dif_data;  // Differential x-section data

        // -----------------------------------------------------------------------
        // MINUIT handling 

        int _print_level   = 0;     // Error code for MINUIT
        int _max_calls     = 1E6;   // Max calls allowed for minimization fcn
        double _tolerance  = 1.E-6; // Minimization tolerance

        ROOT::Math::Minimizer * _minuit;
        ROOT::Math::Functor fcn;

        // Initialize minuit with all our parameter options etc
        void set_up(std::vector<double> starting_guess);

        // -----------------------------------------------------------------------
        // Calcualtions of chi-squared 
        
        // Calculate the chi2 for a given set of parameters, pars, 
        // from a given integrated x-section data set
        double chi2_int(data_set data);

        // Calculate the chi2 for a given set of parameters, pars, 
        // from a given differential x-section data set
        double chi2_dif(data_set data);

        // This is the actual function that gets called by minuit
        // Combined chi2 from all observables and data sets
        double fit_chi2(const double *pars);

        // Save of the last fit run
        bool _fit = false;          // Whether a fit has already been done or not yet
        double _chi2dof, _chi2;     // Last saved chi2/dof
        std::vector<double> _fit_pars, _errors;

        // Save of the best fits found if running multiple times
        double _best_chi2, _best_chi2dof, _best_norm = 1, _best_norm_err;
        std::vector<double> _best_pars, _best_errs;

        // -----------------------------------------------------------------------
        // Parameter handling

        // Store of parameter info
        parameter _norm;
        std::vector<parameter> _pars;

        // Number of free parameters
        int _Nfree = 0;

        // Default guess_range to initalize parameters
        std::array<double,2> _guess_range = {-5, 5};

        // Return the index given a label
        int find_parameter(std::string label);

        // Minuit uses C style arrays to pass parameters. 
        // This method converts them to C++ vectors
        std::vector<double> convert(const double * cpars);

        // Random number generator for creating initial guesses;
        TRandom *_guesser = new TRandom(0);

        // -----------------------------------------------------------------------
        // Methods to print out status to command line

        // Summary of data sets that have been recieved
        void data_info();

        // Similar summary for parameters
        // Display alongside a vector of current parameter values
        // bool start is whether this is the starting guess vector or the 
        // best fit results
        void parameter_info(std::vector<double> guess);

        // After a fit return a summary of fit results
        void print_results(bool last_fit = true);
    };
};

#endif
