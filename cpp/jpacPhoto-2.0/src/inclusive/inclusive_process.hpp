// Extention of the amplitude structure but for inclusive reactions,
// here things are much easier since everything needs to be implements
// at the cross-section level not the amplitude level
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------

#ifndef INCLUSIVE_PROCESS_HPP
#define INCLUSIVE_PROCESS_HPP

#include "constants.hpp"

#include "Math/IntegratorMultiDim.h"
#include "Math/GSLIntegrator.h"
#include "Math/IntegrationTypes.h"
#include "Math/Functor.h"

#include <memory>
#include <functional>
#include <vector>

namespace jpacPhoto
{
    // Forward declare for typedef 
    class raw_inclusive_process;

    // Similar to amplitudes, we only want our inclusive objects to exist as pointers 
    using inclusive_process = std::shared_ptr<raw_inclusive_process>;

    // To make non-pointer objects forbidden we use a key idiom
    class inclusive_key
    {
        private: 
        inclusive_key(){};

        // This function acts as the constructor for an inclusive 
        template<class A>
        friend inclusive_process new_inclusive_process(double, std::string);

        template<class A, class B>
        friend inclusive_process new_inclusive_process(double, B parameter, std::string);
    };

    // Use these functions as our constructor
    template<class A>
    inline inclusive_process new_inclusive_process(double mX, std::string id)
    {
        auto ptr = std::make_shared<A>(inclusive_key(), mX, id);
        return std::static_pointer_cast<raw_inclusive_process>(ptr);
    };

    template<class A, class B>
    inline inclusive_process new_inclusive_process(double mX, B parameter, std::string id)
    {
        auto ptr = std::make_shared<A>(inclusive_key(), mX, parameter, id);
        return std::static_pointer_cast<raw_inclusive_process>(ptr);
    };

    class raw_inclusive_process
    {
        public: 

        // Set both observed particle and target masses
        raw_inclusive_process(inclusive_key key, double mX, std::string id)
        : _mX2(mX*mX), _id(id)
        {};

        // Access internals
        inline double get_observed_mass(){ return sqrt(_mX2); };
        inline double get_target_mass()  { return sqrt(_mT2); };
        inline std::string id(){ return _id; };

        // Set internals
        inline void set_observed_mass(double m){ _mX2 = m*m; };
        inline void set_target_mass(  double m){ _mT2 = m*m; };
        inline void set_id(std::string id){ _id = id; };

        // Every implementation needs to set the minimum missing mass 
        virtual double minimum_M2() = 0;

        // Set the total center of mass energy
        inline void   set_total_energy(double s){ _s = s; };
        inline double get_total_energy(){ return _s; };

        // Number of free parameters
        inline int N_pars(){ return _N_pars; };

        // This function is what a user actually calls
        // It wraps the protected vitual method allocate_parameters() with checks of correct size and caching
        void set_parameters( std::vector<double> x );

        // Given a vector of double of appropriate length, allocate free parameters to model
        // By default we do nothing
        virtual void allocate_parameters( std::vector<double> x ){ return; };

        // Pass a flag and make the appropriate changes, defaults to do nothing excpet save the flag
        virtual inline void set_option( int opt ){ _option = opt; };
        
        // ----------------------------------------------------------------------
        // Kinematics 

        // Momentum of the photon
        double qGamma();

        // Max momentum of the produced particle (exclusive limit)
        double pMax();

        // POLAR VARIABLES (r, cos)
        double EfromRCOS(   double r, double cos);
        double jacobianRCOS(double r, double cos);

        // CARTESIAN VARIABLES (x, y)
        double EfromXY(    double x, double y);
        double jacobianXY( double x, double y);
        double EfromXY2(   double x, double y2);
        double jacobianXY2(double x, double y2);
        double XfromRCOS(  double r, double cos);
        double XfromTM2(   double t, double M2);

        // INVARIANT variables (t, M2)
        double pXfromM2(   double M2);
        double COSfromTM2( double t,  double M2);
        double RfromTM2(   double t,  double M2);
        double jacobianTM2(double t,  double M2);
        double TfromM2COS( double M2, double cos);
        double M2fromRCOS( double r,  double cos);
        double TfromRCOS(  double r,  double cos);
        double TMINfromM2( double M2);
        double TMAXfromM2( double M2);
        double M2MINfromT( double t);
        double M2MAXfromT( double t);

        // MIXED variables (t, x)
        double M2fromXY(  double x, double y);
        double M2fromXY2( double x, double y2);
        double TfromXY(   double x, double y);
        double TfromXY2(  double x, double y2);
        double jacobianTX(double t, double x);
        double M2fromTX(  double t, double x);
        double TMINfromX( double x);
        double TMAXfromX( double x);

        //--------------------------------------------------------------------
        // d3sigma/d3p (invariant cross-section)
        // These need to be specified by a specific parameterization
        // The third argument (double mm) can be either M2 or x
        // which is assumed to correspond to the _useTX member above

    
        // Alias function to rename d3sigma_d3p to more human name
        virtual double invariant_xsection(double s, double t, double mm) = 0;

        // Doubly differential 
        double dsigma_dtdM2(double s, double t, double M2);
        double dsigma_dtdx( double s, double t, double x);
        double dsigma_dxdy2(double s, double x, double y2);
        inline double dsigma_dxdy(double s, double x, double y){ return dsigma_dxdy2(s, x, y*y) / (2.*y); };

        // Integrated cross-sections

        // (t, M2)
        double dsigma_dt( double s, double t);    // integrated over M2
        double dsigma_dM2(double s, double M2);   // integrated over t

        // (x, y2)
        double dsigma_dy2(double s,  double y2);   // integrated over x
        double dsigma_dx( double s,  double x);    // integrated over pT2
        inline double dsigma_dy(double  s,  double y ){ return dsigma_dy2(s, y*y) / (2.*y); };

        // Fully integrated
        double integrated_xsection(double s);   
        
        protected:

        // Int flag used to signal desired changes to amplitude
        int _option = 0;
        
        // Mass of the observed and target particles
        double _mX2 = 0, _mT2 = M2_PROTON;

        // Total center-of-mass energy
        double _s = 100;

        // Different parameterizations may use different variables which make things tricky
        // when integrating. So I include this flag (set to default as false):
        // False -> assume independent variables are t and M2 
        // True  -> assume independent variables are t and x
        inline virtual bool use_TX(){ return false; };   

        // String ID 
        std::string _id = "inclusive_process";

        // Number of parameters
        inline void set_N_pars(int Npars){ _N_pars = Npars; };
        int  _N_pars = 0;

        // Simple check that a given vector is of the expected size
        bool correct_size(std::vector<double> pars);
    };
};

#endif