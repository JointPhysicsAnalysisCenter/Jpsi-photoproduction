// The core object of interest is the amplitude class which encorporates
// the dynamical physics model for a process
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2022)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        dwinney@iu.alumni.edu
// ------------------------------------------------------------------------------

#include "amplitude.hpp"
#include "constants.hpp"
#include "kinematics.hpp"
#include <cstddef>
#include <string>

namespace jpacPhoto
{
    // ------------------------------------------------------------------------------
    // External methods for amplitude sums

    bool are_compatible(amplitude a, amplitude b)
    {
        // Two amplitudes are compatible if they point to the same
        // kinematics object (i.e. are synced together)
        bool same_kinem= (a->_kinematics == b->_kinematics);

        // But also they must have helicites defined in the same frame
        // TODO: add helicity crossing relations so this is not a necessity
        bool same_frame   = (a->native_helicity_frame() == b->native_helicity_frame());

        
        std::string error_msg = "Attempted to add incompatible amplitudes: " + a->id() + " and " + b->id() + "!";
        if (!same_frame) 
        {
            return error(error_msg + " (Helicites defined in different frames!)", false);
        };

        if (!same_kinem)
        {
            return error(error_msg + " (Contain different kinematics objects!)", false);
        };

        return true;
    };
    
    // Sum two amplitudes into a new amplitude 
    amplitude operator+(amplitude a, amplitude b)
    {
        if (!are_compatible(a,b)) return nullptr;
        
        // New id will be the "sum" of the individual id's
        std::string id = a->id() + " + " + b->id(); 

        // Extract the sub amplitudes from each vector being summed
        // This allows the saved pointers to all ways be "base" level amplitudes in stead of other sums
        std::vector<amplitude> from_a = (a->is_sum()) ? a->_subamplitudes : std::vector<amplitude>{{a}};
        std::vector<amplitude> from_b = (b->is_sum()) ? b->_subamplitudes : std::vector<amplitude>{{b}};
        from_a.insert(from_a.end(), from_b.begin(), from_b.end());

        // Initialize and return a new amplitude
        return std::make_shared<raw_amplitude>(amplitude_key(), from_a, id);
    };

    // Take an existing sum and add a new amplitude
    void operator +=(amplitude a, amplitude b)
    {
        if ( !a->is_sum() )
        {
            warning("amplitude::operator+= ", 
                    "Attempting to sum two non-sums! \nPlease initialize a sum of amplitudes with = first (auto c = a + b;) then use the += operator (c += d;)!");
            return;
        };  

        a->add(b);
    };

    // ------------------------------------------------------------------------------
    // Internal methods for amplitude sums

    // Check if a given amplitude is compatible to be summed with *this
    bool raw_amplitude::is_compatible(amplitude new_amp)
    {
        // Two amplitudes are compatible if they point to the same
        // kinematics object (i.e. are synced together)
        bool same_kinem = (new_amp->_kinematics == _kinematics);

        // But also they must have helicites defined in the same frame
        // TODO: add helicity crossing relations so this is not a necessity
        bool same_frame   = (new_amp->native_helicity_frame() == native_helicity_frame());

        
        std::string error_msg = "Attempted to add incompatible amplitudes: " + id() + " and " + new_amp->id() + "!";
        if (!same_frame) 
        {
            return error(error_msg + " (Helicites defined in different frames!)", false);
        };

        if (!same_kinem)
        {
            return error(error_msg + " (Contain different kinematics objects!)", false);
        };
        return true;
    };

    // Add a new amplitude to the list
    void raw_amplitude::add(amplitude new_amp)
    {
        // If our list is empty, capture the first entry we encounter
        if (_subamplitudes.size() == 0)
        {
            _kinematics = new_amp->_kinematics;
            _N_pars     = new_amp->N_pars();
            _subamplitudes.push_back(new_amp);
            return;
        };

        // Each subsequent entry gets compared to the first
        if (is_compatible(new_amp))
        {
            // If new_amp is itself a sum of amps,
            // reach in and pull out the components 
            if (new_amp->is_sum())
            {
                add(new_amp->_subamplitudes);
                return;
            };

            _subamplitudes.push_back(new_amp);
            _N_pars += new_amp->N_pars();
            return;
        };
    };

    // Add a whole vector, this just loops over each entry
    void raw_amplitude::add(std::vector<amplitude> new_amps)
    {
        for (auto amp : new_amps)
        {
            add(amp);
        };
    };

    // ------------------------------------------------------------------------------
    // Updators for the different caches we have

    // Save current kinematic quantities from _kinematics
    void raw_amplitude::store(std::array<int,4> helicities, double s, double t)
    {
        _lamB = helicities[0];
        _lamT = helicities[1];
        _lamX = helicities[2];
        _lamR = helicities[3];

        _s = s; _t = t; _theta = _kinematics->theta_s(s,t);

        _mB = _kinematics->get_beam_mass();
        _mT = _kinematics->get_target_mass();
        _mX = _kinematics->get_meson_mass();
        _mR = _kinematics->get_recoil_mass();

        return;
    };

    // Check for any changes within the set tolerance 
    // And recalculate helicity amplitudes if needed
    void raw_amplitude::update_cache(double s, double t)
    {
        bool s_changed  = !are_equal(_cached_s, s, _cache_tolerance);
        bool t_changed  = !are_equal(_cached_t, t, _cache_tolerance);
        bool mX_changed = !are_equal(_cached_mX, _kinematics->get_meson_mass(),  _cache_tolerance);
        bool mR_changed = !are_equal(_cached_mR, _kinematics->get_recoil_mass(), _cache_tolerance);

        bool need_update = _parameters_changed || s_changed || t_changed || mX_changed || mR_changed;

        if (need_update)
        {
            _cached_helicity_amplitudes.clear();

            // Total number of amplitudes
            int n = (native_helicity_frame() == HELICITY_INDEPENDENT) ? 1 : _kinematics->N_amps();

            if (n == 1)
            {
                // Arbitrary pick the first helicity set to evvaluate, skip parity phase calculations
                _cached_helicity_amplitudes.push_back(helicity_amplitude(_kinematics->helicities(0), s, t));
            }
            else
            {
                // Otherwise we only calculate the first half amplitudes
                // with the rest related by the parity_phase
                for (int i = 0; i < n; i++)
                {
                    complex hel_amp;
                    if (i < n/2) hel_amp = helicity_amplitude(_kinematics->helicities(i), s, t);
                    else         hel_amp = parity_phase(i) * _cached_helicity_amplitudes[n - 1 - i];
                    _cached_helicity_amplitudes.push_back(hel_amp); 
                }
            };

            if (_cached_helicity_amplitudes.size() != n)
            {
                warning(id()+"::update_cache", "Cached size doesn't match expected number of helicity amplitude!");
            };
        };

        // Update the cache info as well
        _cached_mX = _kinematics->get_meson_mass();
        _cached_mR = _kinematics->get_recoil_mass();
        _cached_s  = s; _cached_t = t;
        _parameters_changed = false;
    };

    // ------------------------------------------------------------------------------
    // Phase properties of helicity amplitudes

    double raw_amplitude::parity_phase(std::array<int,4> helicities)
    {
        return _kinematics->parity_phase(helicities, this->native_helicity_frame());
    };

    double raw_amplitude::parity_phase(int i)
    {
        return _kinematics->parity_phase(i, this->native_helicity_frame());
    };

    // ------------------------------------------------------------------------------
    // Parameter handling

    // Ability to change number of expected parameters from inside a class
    void raw_amplitude::set_N_pars(int N)
    {
        _N_pars = N;
    };

    // Simple check that a given vector is of the expected size
    bool raw_amplitude::correct_size(std::vector<double> pars)
    {
        if (pars.size() != _N_pars)
        {
            return error(id()+"::set_parameters", "Number of parameters passed not the expected size!", false);
        };
        return true;
    };

    // ------------------------------------------------------------------------------
    // Compatability checks

    // At construction, check the kinematics with set quantum numbers is compatable
    void raw_amplitude::check_QNs(kinematics kinem)
    {
        // Get all the allowed JP's from the amplitude
        std::vector<particle> allowed_mesons, allowed_baryons;
        allowed_mesons  = this->allowed_mesons();
        allowed_baryons = this->allowed_baryons();

        // Grab the requested JPs from the raw_kinematics
        particle requested_meson, requested_baryon;
        requested_meson  = kinem->get_meson();
        requested_baryon = kinem->get_baryon();

        // Check if they are allowed
        bool meson_fails, baryon_fails;
        meson_fails  = std::find(allowed_mesons.begin(),  allowed_mesons.end(),  requested_meson)  == allowed_mesons.end();
        baryon_fails = std::find(allowed_baryons.begin(), allowed_baryons.end(), requested_baryon) == allowed_baryons.end();
    
        auto requested_meson_JP = kinem->get_meson_JP();
        auto requested_baryon_JP = kinem->get_meson_JP();

        if (meson_fails)
        {
            warning(id()+"::check_QNs", "Requested meson quantum numbers (J=" + std::to_string(requested_meson_JP[0]) + ", P=" + std::to_string(requested_meson_JP[1])+") not available!");
        };
    
        if (baryon_fails)
        {
            warning(id()+"::check_QNs", "Requested baryon quantum numbers (J=" + std::to_string(requested_baryon_JP[0]) + "/2, P=" + std::to_string(requested_baryon_JP[1])+") not available!");
        };    
    };

    // ------------------------------------------------------------------------------
    // OBSERVABLES
    
    // Square of the spin averaged amplitude squared
    double raw_amplitude::probability_distribution(double s, double t)
    {
        // Check we have the right amplitudes cached
        update_cache(s, t);

        double sum = 0.;
        for (auto amp : _cached_helicity_amplitudes)
        {
            sum += std::norm(amp);
        };

        return sum;
    };

    // Differential cross section dsigma / dt
    // in NANOBARN
    double raw_amplitude::differential_xsection(double s, double t)
    {
        if (s < _kinematics->sth()) return 0.;

        double sum = probability_distribution(s, t);

        double norm = 1.;
        norm /= 64. * PI * s;
        norm /= std::real(pow(_kinematics->initial_momentum(s), 2.));
        norm /= (2.56819E-6); // Convert from GeV^-2 -> nb

        // Average over initial helicities
        if (native_helicity_frame() !=  HELICITY_INDEPENDENT)
        {
            norm /= 4*(_kinematics->is_photon()) + 6*(!_kinematics->is_photon());
        }

        return norm * sum;
    };

    // Integrated total cross-section
    // IN NANOBARN
    double raw_amplitude::integrated_xsection(double s)
    {
        if (s < _kinematics->sth()) return 0.;

        auto F = [&](double t)
        {
            return differential_xsection(s, t);
        };

        ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE, ROOT::Math::Integration::kGAUSS61);
        ROOT::Math::Functor1D wF(F);
        ig.SetFunction(wF);

        double t_min = _kinematics->t_min(s);
        double t_max = _kinematics->t_max(s);

        return ig.Integral(t_max, t_min);
    };

    // Polarization asymmetry between beam and recoil baryon
    double raw_amplitude::K_LL(double s, double t)
    {
        // denominator is the full sum of amps squared;
        double norm = probability_distribution(s, t);

        double sum = 0;
        for (int i = 0; i < _kinematics->N_amps(); i++)
        {
            std::array<int,4> hel = _kinematics->helicities(i);
            int eta = (1 - hel[0]*hel[3] / abs(hel[0]*hel[3])) / 2;

            complex amp_i = _cached_helicity_amplitudes[i];
            sum += pow(-1, eta) * std::norm(amp_i);
        }

        return sum / norm;
    };

    // Polarization asymmetry between beam and target proton
    double raw_amplitude::A_LL(double s, double t)
    {
        // Calculate the denominator
        // This will also update the cache 
        double norm = probability_distribution(s, t);

        double sum = 0.;
        for (int i = 0; i < _kinematics->N_amps(); i++)
        {
            std::array<int,4> hel = _kinematics->helicities(i);
            int eta = (1 - hel[0]*hel[1] / abs(hel[0]*hel[1])) / 2;

            complex amp_i = _cached_helicity_amplitudes[i];
            sum += pow(-1, eta) * std::norm(amp_i);
        }

        return sum / norm;
    };

    // Photon spin-density matrix elements
    complex raw_amplitude::SDME(int alpha, int lam, int lamp, double s, double t)
    {
        if (alpha < 0 || alpha > 2 || std::abs(lam) > 2 || std::abs(lamp) > 2)
        {
            return error("SDME", "Invalid parameter passed. Returning 0!", 0);
        };

        if (!_kinematics->is_photon()) 
        {
            return error("SDME", "Currently only massless beam is available. Returning 0!", 0);
        };

        // If spin is too small return 0 automatically
        int j = _kinematics->get_meson_JP()[0];
        if (j < 2 && (abs(lam) == 2 || abs(lamp) == 2)) return 0.;
        if (j < 1 && (abs(lam) >= 1 || abs(lamp) >= 1)) return 0.;

        // Check we have the right amplitudes cached
        update_cache(s, t);

        // Phase and whether to conjugate at the end
        bool CONJ = false;
        double phase = 1.;

        // if first index smaller, switch them
        if (std::abs(lam) < std::abs(lamp))
        {
            int temp = lam;
            lam  = lamp;
            lamp = temp;

            CONJ = true;
        }

        // if first index is negative, flip to positive
        if (lam < 0)
        {
            lam  *= -1;
            lamp *= -1;

            phase *= pow(-1., double(lam - lamp));

            if (alpha == 2){phase *= -1.;};
        }

        // Normalization (sum over all amplitudes squared)
        double norm = probability_distribution(s, t);

        // iters[0] keeps track of helicity = + amplitudes
        // while iters[1] keeps track of helicity = - 
        std::array<std::vector<int>, 2> iters = get_iters(j, _kinematics->get_baryon_JP()[0]);
        
        // choose a starting point depending on alpha
        bool pos_or_neg;
        (alpha == 0) ? (pos_or_neg = 0) : (pos_or_neg = 1);

        // k filters first index to be  0, 1, 2
        // l filters second index to be 0, 1, 2
        // m filters sign of second index
        int k, l, m;
        int jlamlamp = 100*j + 10 * lam + abs(lamp);
        switch(jlamlamp)
        {
            case   0: {k =  0; l =  0; break;};
            case 100: {k =  2; l =  2; break;};
            case 110: {k =  0; l =  2; break;};
            case 111: {k =  0; l =  0; break;};
            case 220: {k = -2; l =  2; break;};
            case 221: {k = -2; l =  0; break;};
            case 222: {k = -2; l = -2; break;};
            default: exit(0);
        };
        
        (lamp < 0) ? (m = 4 * j) : (m = 0);

        // Sum over the appropriate amplitude combinations
        complex result = 0.;
        for (int i = 0; i < iters[0].size(); i++)
        {
            int index = iters[pos_or_neg][i];

            complex amp, amp_star, temp;
            amp      = _cached_helicity_amplitudes[index + k];
            amp_star = _cached_helicity_amplitudes[iters[0][i] + l + m];
            
            temp = real(amp * conj(amp_star));
            if (alpha == 2)
            {
                temp *= I * double(_kinematics->helicities(iters[0][i] + k)[0]);
            }
            result += temp;
        }

        if (CONJ == true)
        {
            result = conj(result);
        }

        result /= norm;
        result *= phase;

        return result;
    };

    
    complex raw_amplitude::rotated_SDME(int alpha, int lam, int lamp, double s, double t, double theta)
    {
        // First get the spin of the particle we're rotating
        int J = _kinematics->get_meson_JP()[0];

        // Sum over all the rest
        complex result = 0.;
        for (int m = 0; m <= J; m++)
        {
            for (int mp = 0; mp <= J; mp++)
            {
                result += wigner_d_int(J, lam,  m, theta) * SDME(alpha,  m,  mp, s, t) * wigner_d_int(J, lamp,  mp, theta);
                if (m  != 0) result += wigner_d_int(J, lam, -m, theta) * SDME(alpha, -m,  mp, s, t) * wigner_d_int(J, lamp,  mp, theta);
                if (mp != 0) result += wigner_d_int(J, lam, -m, theta) * SDME(alpha, -m, -mp, s, t) * wigner_d_int(J, lamp, -mp, theta);
                if (m  != 0 && mp != 0) result += wigner_d_int(J, lam,  m, theta) * SDME(alpha,  m, -mp, s, t) * wigner_d_int(J, lamp, -mp, theta);
            };
        }

        return result;
    };

    // Calculate the Helicity frame SDME
    complex raw_amplitude::SDME_H(int alpha, int lam, int lamp, double s, double t)
    {
        helicity_frame frame = this->native_helicity_frame();
        
        switch (frame)
        {
            case S_CHANNEL: return SDME(alpha, lam, lamp, s, t);
            case T_CHANNEL: return rotated_SDME(alpha, lam, lamp, s, t, -_kinematics->H_to_GJ_angle(s, t));
            case U_CHANNEL: 
            {
                return error("SDME_H", "Rotations from u-channel CM frame to Helicty frame not yet implemented... Returning 0.", 0);
            };
        };

        return 0;
    };

    // Calculate the Gottfried-Jackson SDME
    complex raw_amplitude::SDME_GJ(int alpha, int lam, int lamp, double s, double t)
    {
        helicity_frame frame = this->native_helicity_frame();

        switch (frame)
        {
            case S_CHANNEL: return rotated_SDME(alpha, lam, lamp, s, t, _kinematics->H_to_GJ_angle(s, t));
            case T_CHANNEL: return SDME(alpha, lam, lamp, s, t);
            case U_CHANNEL: 
            {
                return error("SDME_GJ", "Rotations from u-channel CM frame to Gottfried-Jackson frame not yet implemented... Returning 0.", 0);
            };
        };

        return 0.;
    };

    // Integrated beam asymmetry sigma_4pi
    double raw_amplitude::beam_asymmetry_4pi(double s, double t)
    {
        // Check we have the right amplitudes cached
        update_cache(s, t);

        double rho100 = real(SDME(1, 0, 0, s, t));
        double rho111 = real(SDME(1, 1, 1, s, t));
        double rho122 = real(SDME(1, 2, 2, s, t));
        double rho000 = real(SDME(0, 0, 0, s, t));
        double rho011 = real(SDME(0, 1, 1, s, t));
        double rho022 = real(SDME(0, 2, 2, s, t));
        
        return -(rho100 + 2 * rho111 + 2 * rho122) / (rho000 + 2 * rho011 + 2 * rho022);
    };

    // Beam asymmetry along y axis sigma_y 
    double raw_amplitude::beam_asymmetry_y(double s, double t)
    {
        // Check we have the right amplitudes cached
        update_cache(s, t);

        double rho111  = real(SDME(1, 1,  1, s, t));
        double rho11m1 = real(SDME(1, 1, -1, s, t));
        double rho011  = real(SDME(0, 1,  1, s, t));
        double rho01m1 = real(SDME(0, 1, -1, s, t));

        return (rho111 + rho11m1) / (rho011 + rho01m1);
    };

    // Parity asymmetry P_sigma
    double raw_amplitude::parity_asymmetry(double s, double t)
    {
        // Check we have the right amplitudes cached
        update_cache(s, t);

        double rho100  = real(SDME(1, 0,  0, s, t));
        double rho11m1 = real(SDME(1, 1, -1, s, t));
        double rho12m2 = real(SDME(1, 2, -2, s, t));

        return 2 * rho11m1 - 2 * rho12m2 - rho100;
    };

};