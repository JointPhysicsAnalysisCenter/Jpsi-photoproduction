// Class which allows an amplitude to be fit to data based on chi2 minimization
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#include "fitter.hpp"
#include "data_set.hpp"

namespace jpacPhoto
{
    // -----------------------------------------------------------------------
    // Methods for managing data

    void fitter::add_data(data_set data)
    {
        switch (data._type)
        {
            case integrated_data:   _int_data.push_back(data); break;
            case differential_data: _dif_data.push_back(data); break;
            default:
            {
                warning("fitter::add_data", "data_set " + data._id + " of unsupported type!");
                return;
            }
        }

        _N += data._N;
    };

    void fitter::clear_data()
    {
        _N = 0;
        _int_data.clear();
        _dif_data.clear();
    };

    // -----------------------------------------------------------------------
    // Set limits, labels, and fix parameters

    void fitter::reset_parameters()
    {
        _pars.clear();
        _Nfree = _amplitude->N_pars();

        // populate parameters vector of appropriate size
        for (int i = 0; i < _amplitude->N_pars(); i++)
        {
            _pars.push_back(i);
        };
        set_parameter_labels(_amplitude->parameter_labels());
    };

    // Give each parameter a label beyond their default par[i] name
    void fitter::set_parameter_labels(std::vector<std::string> labels)
    {
        if (labels.size() != _pars.size())
        {
            warning("fitter::set_parameter_labels", "Labels vector does not match number of parameters!");
            return;
        }

        for (int i = 0; i < _pars.size(); i++)
        {
            _pars[i]._label = labels[i];
        };
    };

    // Given a parameter label, find the corresponding index
    int fitter::find_parameter(std::string label)
    {
        for (auto par : _pars)
        {
            if (par._label == label) return par._i;
        };

        if (label == "normalization" || label == "norm") return -1;

        return error("fitter::find_parameter", "Cannot find parameter labeled " + label + "!", -2);
    };

    // Set limits and/or a custom stepsize
    void fitter::set_parameter_limits(parameter& par, std::array<double,2> bounds, double step)
    {
        par._custom_limits = true;
        par._lower         = bounds[0];
        par._upper         = bounds[1];
        par._step          = step;
    };

    void fitter::set_parameter_limits(std::string label, std::array<double,2> bounds, double step)
    {
        int index = find_parameter(label);
        if (index == -2) return;
        if (index == -1) return set_parameter_limits(_norm, bounds, step);
        return set_parameter_limits(_pars[index], bounds, step);
    }

    void fitter::fix_parameter(parameter& par, double val)
    {
        // If parameter is already fixed, just update the fixed val
        // otherwise flip the fixed flag and update the number of free pars
        if (!par._fixed && !par._is_norm) _Nfree--;

        par._fixed = true;
        par._value = val;
        _fit = false;
    };

    void fitter::fix_parameter(std::string label, double val)
    {
        int index = find_parameter(label);
        if (index == -2) return;
        if (index == -1) return fix_parameter(_norm, val);
        return fix_parameter(_pars[index], val);
    };

    void fitter::free_parameter(parameter& par)
    {
        // if not fixed, this does nothing
        if (!par._fixed) return;

        par._fixed = false;
        _fit = false;

        if (!par._is_norm) _Nfree++;
    };

    void fitter::free_parameter(std::string label)
    {
        int index = find_parameter(label);
        if (index == -1) return free_parameter(_norm);
        return free_parameter(_pars[index]);
    };

    // Given a C-style array of size _Nfree
    // Convert to a C++ style std::vector and populate
    // fixed value parameters in the expected order
    std::vector<double> fitter::convert(const double * cpars)
    {
        std::vector<double> result;

        // Move along the pars index when a parameter is not fixed
        int i = 0;
        for (auto par : _pars)
        {
            if (par._fixed)
            {
                result.push_back(par._value);
            }
            else 
            {
                result.push_back(cpars[i]);
                i++;
            }
        };

        if (i != _Nfree) warning("fitter::convert", "Something went wrong in converting parameter vector.");
        return result;
    };

    // ---------------------------------------------------------------------------
    // Chi-squared functions

    // Total chi2, this combines all data sets and is the function that gets called 
    // by the minimizer
    double fitter::fit_chi2(const double * cpars)
    {
        // IF the norm is being fit, we'll use the 
        if (!_norm._fixed) _norm._value = cpars[_Nfree];

        // First convert the C string to a C++ vector
        std::vector<double> pars = convert(cpars);

        // Pass parameters to the amplitude
        _amplitude->set_parameters(pars);

        // Then sum over all data sets and observables
        double chi2 = 0;
        for (auto data : _int_data)
        {
            chi2 += chi2_int(data);
        };
        for (auto data : _dif_data)
        {
            chi2 += chi2_dif(data);
        };

        return chi2;
    };

    // Calculate the chi2 from integrated xsection data
    double fitter::chi2_int(data_set data)
    {
        // Sum over data points
        double chi2 = 0;
        for (int i = 0; i < data._N; i++)
        {
            double W = (data._lab) ? W_cm(data._w[i]) : data._w[i];
            double s = W*W;

            double sigma_th = pow(_norm._value, 2) * _amplitude->integrated_xsection(s);
            double sigma_ex = data._obs[i];
            double error    = data._obserr[i];

            chi2 += pow((sigma_th - sigma_ex)/error, 2);
        };

        return chi2;
    };

    // Calculate chi2 from differential data set
    double fitter::chi2_dif(data_set data)
    {   
        double chi2 = 0;
        for (int i = 0; i < data._N; i++)
        {
            double W = (data._lab) ? W_cm(data._w[i]) : data._w[i];
            double s = W*W;

            double t = (data._negt) ? -data._t[i] : data._t[i];
            if (data._tprime) t += _amplitude->_kinematics->t_min(s);

            double sigma_th = pow(_norm._value, 2) * _amplitude->differential_xsection(s, t);
            double sigma_ex = data._obs[i];
            double error    = data._obserr[i];

            chi2 += pow((sigma_th - sigma_ex)/error, 2);
        };

        return chi2;
    };
    
    // ---------------------------------------------------------------------------
    // Load parameter information into minuit

    void fitter::set_up(std::vector<double> starting_guess)
    {
        _minuit->Clear();
        _minuit->SetTolerance(_tolerance);
        _minuit->SetPrintLevel(_print_level);
        _minuit->SetMaxFunctionCalls(_max_calls);

        // Iterate over each _par but also keep track of the index in starting_guess
        int i = 0;
        for ( auto par : _pars)
        {   
            if (par._fixed) continue;
            
            _minuit->SetVariable(i, par._label, starting_guess[i], par._step);

            if (par._custom_limits) _minuit->SetVariableLimits(i, par._lower, par._upper);

            // If we've made it this far, this par isnt fixed, and we move to the next index 
            i++;
        };

        // AT the end add the normalization if this has not been fixed
        // If normalization is being fixed, always initialize it at starting value = 1
        if (!_norm._fixed) 
        {
            _minuit->SetLowerLimitedVariable(_Nfree, "norm", 1., _norm._step, 0.);
        };
    
        fcn = ROOT::Math::Functor(this, &fitter::fit_chi2, _Nfree + !_norm._fixed);
        _minuit->SetFunction(fcn);
    };

    // Actually do the fit given a vector of size amp->N_pars() as starting values
    // Prints results to command line and sets best fit parameters into _amplitude
    void fitter::do_fit(std::vector<double> starting_guess, bool show_data)
    {
        if (starting_guess.size() != _Nfree) 
        {
            warning("fitter::do_fit", "Starting guess not the correct size! Expected " + std::to_string(_Nfree) + " parameters!");
            return;
        };

        set_up(starting_guess);

        if (show_data) { line(); data_info(); };
        parameter_info(starting_guess);


        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Beginning fit..." << std::flush; 

        if (_print_level != 0) line();   
        _minuit->Minimize();
        if (_print_level != 0) line();   

        std::cout << "Done! \n";

        // Timing info
        auto stop     = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast< std::chrono::seconds>(stop - start);
        std::cout << std::left << "Finished in " << duration.count() << " s" << std::endl;

        line();
        print_results();
    };

    // Repeat do_fit N times and find the best fit
    // Parameters are randomly initialized each time on the interval [-5, 5] unless custom limits are set
    void fitter::do_fit(int N)
    {
        divider();
        std::cout << std::left << "Commencing N = " + std::to_string(N) + " fit iterations." << std::endl;

        // Initial guess
        std::vector<double> guess;

        for (int i = 1; i <= N; i++)
        {
            guess.clear();

            // Whether this is the first iteration
            bool first_fit =  !(i-1);
            
            // Initialize the guess for each parameter
            for (auto par : _pars)
            {
                if (par._fixed) continue;

                if (par._custom_limits) guess.push_back(_guesser->Uniform(par._lower, par._upper)); 
                else                    guess.push_back(_guesser->Uniform(_guess_range[0], _guess_range[1]));
            };

            // Do out fit with this random guess
            if (!first_fit) std::cout << std::left << "Fit (" + std::to_string(i) + "/" + std::to_string(N) + ")" << std::endl;
            do_fit(guess, first_fit);

            // Compare with previous best and update
            if ( first_fit || chi2dof() < _best_chi2dof)
            {
                _best_chi2dof = _minuit->MinValue() / (_N - _minuit->NFree());
                _best_chi2    = _minuit->MinValue();
                _best_pars    = convert(_minuit->X());
                _best_errs    = convert(_minuit->Errors());

                if (!_norm._fixed)
                {
                    _best_norm     = _minuit->X()[_Nfree];
                    _best_norm_err = _minuit->Errors()[_Nfree];
                };
            };
        };

        // After looping, set the best_pars to the amplitude
        _amplitude->set_parameters(_best_pars);

        // And set the global saved pars to the best_fit
        std::cout << std::left << "Best fit found after N = " + std::to_string(N) + " iterations" << std::endl;
        line();
        print_results(false);
    };

    // ---------------------------------------------------------------------------
    // Status messages printed to command line

    // Summary of data sets that have been recieved
    void fitter::data_info()
    {
        using std::cout; 
        using std::left;
        using std::setw;
        using std::endl;

        cout << left;
        divider();
        cout << "Fitting amplitude (\"" << _amplitude->id() << "\") to " << _N << " data points:" << endl;
        line();
        cout << setw(30) << "DATA SET"         << setw(20) << "OBSERVABLE"     << setw(10) << "POINTS" << endl;
        cout << setw(30) << "----------------" << setw(20) << "--------------" << setw(10) << "-------" << endl;
        for (auto data : _int_data)
        {
            cout << setw(30) << data._id  << setw(20)  << "integrated xs"   << setw(10) << data._w.size()  << endl;  
        };
        for (auto data : _dif_data)
        {   
            cout << setw(30) << data._id  << setw(20) << "differential xs" << setw(10) << data._t.size() << endl;  
        };
    };

    // Print out a little table of the starting values of all the parameters and other set options
    // Here starting_guess should be of size _Nfree!
    void fitter::parameter_info(std::vector<double> starting_guess)
    {  
        using std::cout; 
        using std::left;
        using std::setw;
        using std::endl;

        cout << std::setprecision(8);
        cout << left;

        line(); divider();
        // Print message at the beginning of the fit
        cout << "Fitting " + std::to_string(_Nfree) << " (of " << std::to_string(_pars.size()) << ") parameters";

        if (!_norm._fixed) cout << " and an overall normalization: " << endl; 
        else               cout << ":" << endl;

        line();

        cout << left << setw(10) << "N"     << setw(17) << "PARAMETER"  << setw(20) << "START VALUE"  << endl;
        cout << left << setw(10) << "-----" << setw(17) << "----------" << setw(20) << "------------" << endl;

        if (!_norm._fixed)
        {
            cout << left << setw(10) << "-"     << setw(17) << "Norm."  << setw(20) << "1"  << endl;
        };

        // Moving index from the guess vector
        int i = 0;
        for (auto par : _pars)
        {
            // Parse whether a parameter has extra options 
            // such as custom limits
            std::string extra = "";
            if (par._custom_limits)
            {   
                std::stringstream ss;
                ss << std::setprecision(5) << "[" << par._lower << ", " << par._upper << "]";
                extra = ss.str();
            };

            // Or is fixed
            double par_val;
            if (par._fixed)
            {
                par_val = par._value;
                extra   = "[FIXED]";
            }
            else
            {
                par_val = starting_guess[i];
                i++;
            }

            cout << left << setw(10) << par._i << setw(17) << par._label << setw(20) << par_val << setw(20) << extra << endl;
        };
        line(); divider(); line();
    };

    // At the end of a fit, print out a table sumarizing the fit results
    // if last_fit == true, we grab the results from the most recent fit in _minuit
    // else we print out the ones saved in _best_fit
    void fitter::print_results(bool last_fit)
    {
        using std::cout; 
        using std::left;
        using std::setw;
        using std::endl;

        cout << std::setprecision(8);
        cout << left;

        int dof                  = _N - _minuit->NFree();
        double chi2              = (last_fit) ? _minuit->MinValue()               : _best_chi2;
        double chi2dof           = (last_fit) ? _minuit->MinValue() / double(dof) : _best_chi2dof;
        std::vector<double> pars = (last_fit) ? convert(_minuit->X())             : _best_pars;
        std::vector<double> errs = (last_fit) ? convert(_minuit->Errors())        : _best_errs;


        divider();
        std::cout << std::left << std::setw(10) << "chi2 = "      << std::setw(15) << chi2;
        std::cout << std::left << std::setw(10) << "chi2/dof = "  << std::setw(15) << chi2dof << "\n";

        line();

        cout << left << setw(10) << "N"     << setw(16) << "PARAMETER"  << setw(18) << "FIT VALUE"    << setw(18) << "ERROR"        << endl;
        cout << left << setw(10) << "-----" << setw(16) << "----------" << setw(18) << "------------" << setw(18) << "------------" << endl;

        if (!_norm._fixed) 
        {
            double norm     = (last_fit) ? _minuit->X()[_Nfree]      : _best_norm;
            double norm_err = (last_fit) ? _minuit->Errors()[_Nfree] : _best_norm_err; 
            cout << left << setw(10) << "-"     << setw(16) << "Norm."  << setw(18) << norm << setw(18) << norm_err << endl;
        };

        for (auto par : _pars)
        {
            double val;
            std::string err;
            std::stringstream ss;
            ss << std::setprecision(8);

            if (par._fixed)
            {
                val = par._value;
                err = "[FIXED]";
            }
            else
            {
                val = pars[par._i];
                ss << errs[par._i];
                err = ss.str();
            }

            cout << left << setw(10) << par._i << setw(16) << par._label << setw(18) << val << setw(18) << err << endl;
        };
        line(); divider(); line();
        
        // At the end update the amplitude parameters to include the fit results
        _amplitude->set_parameters(pars);

        _chi2     = chi2;
        _chi2dof  = chi2dof;
        _fit_pars = pars;

        // Let rest of the fitter that a fit result has been saved
        _fit = true;
    }
};