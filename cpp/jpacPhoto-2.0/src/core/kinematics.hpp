// Class to contain all relevant kinematic quantities. 
// (e.g. angles, momenta, helicity info)
//
// Full kinematics of the process are entirely determiend by the masses of all particles:
// mB - beam mass 
// mT - target (baryon) mass
// mX - produced meson mass
// mR - recoil baryon mass
//
// As well as quantum numbers of the final states
// All of these are allowed to float with which combinations are allowed handled by individual amplitudes
// 
// --------------------------------------------------------------------------------
// Author:       Daniel Winney (2020)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@iu.edu
// --------------------------------------------------------------------------------

#ifndef KINEMATICS_HPP
#define KINEMATICS_HPP

#include "constants.hpp"
#include "helicities.hpp"

#include "TMath.h"

#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

namespace jpacPhoto
{
    // ---------------------------------------------------------------------------
    // The reaction kinematics object is intended to have all relevant kinematic quantities
    // forthe reaction. Here you'll find the momenta and energies of all particles, angles,
    // invariant, etc.
    //
    // These are the basis for more complicated structures such as covariants or the
    // amplitudes themselves
    // ---------------------------------------------------------------------------
    
    // Forward declaration so we can rename ptr to kinematics as just kinematics
    // WE do this because we basically never want to work with a raw instance, but pass around a pointer
    class raw_kinematics;
    using kinematics = std::shared_ptr<jpacPhoto::raw_kinematics>;

    // Constructors are made private to prevent raw_kinematics objects from being initialized
    // All user interactions should be through the pointer class kinematics
    class masses
    {
        private:
        
        friend kinematics new_kinematics(double, double);
        friend kinematics new_kinematics(double, double, double, double);

        explicit masses(std::array<double,4> m)
        : _mB(m[0]), _mB2(m[0]*m[0]), _mT(m[1]), _mT2(m[1]*m[1]),
          _mX(m[2]), _mX2(m[2]*m[2]), _mR(m[3]), _mR2(m[3]*m[3])
        {};

        public:

        double _mB = 0.,       _mB2 = 0.;         // mass and mass squared of the "beam" 
        double _mX = 0.,       _mX2 = 0.;         // mass and mass squared of the produced particle
        double _mT = M_PROTON, _mT2 = M2_PROTON;  // mass of the target, assumed to be proton unless overriden
        double _mR = M_PROTON, _mR2 = M2_PROTON;  // mass of the recoil baryon, assumed to be proton unless overriden
    };

    // With array<double,4> this specifies all four masses in order
    // {beam, target, produced, recoil}
    inline kinematics new_kinematics(double m1, double m2, double m3, double m4)
    {
        return std::make_shared<raw_kinematics>(masses{{m1, m2, m3, m4}});
    };
    
    // // If only size 2 array is passes this specifies final state
    // // and initial state is assumed to be photon + proton
    inline kinematics new_kinematics(double m1, double m2 = M_PROTON)
    {
        return std::make_shared<raw_kinematics>(masses{{0., M_PROTON, m1, m2}});
    };

    class raw_kinematics
    {
        public: 

        // -----------------------------------------------------------------------
        // Only constructor requires the masses struct which is private
        // This is to prevent raw_kinematics objects from being created on the stack
        // We only want kinematics pointer objects to exist.

        raw_kinematics(masses m)
        : _masses(m),  _photon(!(m._mB>0.))
        {};

        // Delete copy and move operators so this is only passable by pointer
        raw_kinematics(const raw_kinematics&) = delete;
        const raw_kinematics &operator = (const raw_kinematics &) = delete;

        // The real "constructor" should be this factory method here.
        // This outputs shared_pointer objects which then are passed around

        // -----------------------------------------------------------------------
        // Accessors and setters for masses 

        inline double Wth(){ return (_masses._mX + _masses._mR); }; // square root of the threshold
        inline double sth(){ return Wth() * Wth(); }; // final state threshold

        // Assessor functions for masses
        inline double get_meson_mass() { return _masses._mX; };
        inline double get_recoil_mass(){ return _masses._mR; };
        inline double get_target_mass(){ return _masses._mT; };
        inline double get_beam_mass()  { return _masses._mB; };
        
        // Setters for masses
        inline void set_meson_mass(double x) { _masses._mX = x; _masses._mX2 = x*x; };
        inline void set_target_mass(double x){ _masses._mT = x; _masses._mT2 = x*x; };
        inline void set_recoil_mass(double x){ _masses._mR = x; _masses._mR2 = x*x; };
        inline void set_beam_mass(double x)  { _masses._mB = x; _masses._mB2 = x*x; };

        // Get whether current kinematics has photon beam
        inline bool is_photon() { return _photon;  };
        inline bool is_virtual(){ return _virtual; };
        inline void set_Q2(double x)
        {
            if (!is_photon()) 
            {
                error("kinematics", "call to set_Q2() without initializing a photon beam first!");
            }

            _virtual = true; 
            set_beam_mass(x);
        }

        // ---------------------------------------------------------------------------
        // Quantum numbers of produced meson. 
        
        inline std::array<int,2> get_meson_JP(){ return _mjp; };
        particle get_meson();

        inline void set_meson_JP(int J, int P)
        { 
            _mjp = {J, P};
            _helicities = get_helicities(J, get_baryon_JP()[0], _photon);
            _nAmps = _helicities.size();
        };
        inline void set_meson_JP(std::array<int,2> jp){ set_meson_JP(jp[0], jp[1]); };
        void set_meson_JP(particle x);
        
        // ---------------------------------------------------------------------------
        // Quantum numbers of produced baryon.
        // The baryon spin (and only this quantity) is multiplied by 2 to be saves as an int

        inline std::array<int,2> get_baryon_JP(){ return _bjp; };
        particle get_baryon();


        inline void set_baryon_JP(int J, int P)
        { 
            _bjp = {J, P};
            _helicities = get_helicities(get_meson_JP()[0], J, _photon);
            _nAmps = _helicities.size();
        };
        inline void set_baryon_JP(std::array<int,2> jp){ set_baryon_JP(jp[0], jp[1]); };
        void set_baryon_JP(particle x);

        // ---------------------------------------------------------------------------
        // Accessing the helicity combinations 
        inline int N_amps(){ return _nAmps; };
        inline int helicity_index(std::array<int,4> hel){ return find_helicity(hel, _mjp[0], _bjp[0], _photon); };
        std::array<int, 4> helicities(int i);
        inline std::vector<std::array<int,4>> helicities(){ return _helicities; };

        //--------------------------------------------------------------------------
        // Other quantities

        // Moduli of the initial and final state 3-momenta.
        double initial_momentum(double s);
        double final_momentum(double s);

        // Energies of all the particles
        double beam_energy(double s);
        double target_energy(double s);
        double meson_energy(double s);
        double recoil_energy(double s);

        // Get s-channel scattering angle from invariants
        double z_s(double s, double t);

        // Scattering angle in the s-channel
        // Use TMath::ACos instead of std::acos because its safer at the end points
        double theta_s(double s, double t);

        // Invariant variables
        double t_man(double s, double theta);
        double u_man(double s, double theta);

        // Bound of cross-variables
        double t_min(double s){ return t_man(s, 0.); };
        double t_max(double s){ return t_man(s, PI); };
        double u_min(double s){ return u_man(s, PI); };
        double u_max(double s){ return u_man(s, 0.); };

        // Scattering angles in t and u channel frames
        double z_t(double s, double theta);
        double z_u(double s, double theta);

        // modulous of 3-momenta in t-channel frame
        complex initial_momentum_tframe(double t);
        complex final_momentum_tframe(double t);
       
        // Intrisnic parity obeyed by helicity amplitude 
        // This depends on which scattering chan we look at and quantum numbers of all particles
        int intrinsic_parity(helicity_frame channel);

        // Phase relating lambda_gamma = +1 and lambda_gamma = -1 amplitudes 
        // Depends on the channel with respect to which the helicities are defined
        int parity_phase(std::array<int, 4> helicities, helicity_frame channel);
        inline int parity_phase(int i, helicity_frame channel)
        {
            return parity_phase(_helicities[i], channel);
        };

        // Wigner rotation angle connecting helicity and gottfried-jackson frames 
        double H_to_GJ_angle(double s, double t);

        // -----------------------------------------------------------------------
        private:

        // Container class holding all the masses of the system
        // Additionally serves as a pass-key class in the constructor
        masses _masses;

        // Masses are private to prevent them from being changed mid calculation 
        // Instead should be manipulated with public accessor and settor above
        bool _photon = true, _virtual = false;    // whether we have a photon and if its virtual

        // Quantum numbers of final state meson and baryon 
        std::array<int,2> _mjp{{1,1}};
        std::array<int,2> _bjp{{1,1}}; // Baryon is multiplied by two (J,P) = (1,1) -> 1/2+

        // Helicity configurations
        // Defaults to spin-1 meson, spin-1/2 baryon
        // Beam [0], Target [1], Produced Meson [2], Recoil Baryon [3]
        int _nAmps = SPIN_ONE_HELICITIES.size(); 
        std::vector< std::array<int, 4> > _helicities = SPIN_ONE_HELICITIES;
    };
};

#endif
