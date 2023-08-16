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

#include "inclusive_process.hpp"

namespace jpacPhoto
{
    bool raw_inclusive_process::correct_size(std::vector<double> pars)
    {
        if (pars.size() != _N_pars)
        {
            return error(id()+"::set_parameters", "Number of parameters passed not the expected size!", false);
        };
        return true;
    };

    void raw_inclusive_process::set_parameters( std::vector<double> x )
    {
        // Check the size is alright
        if (!correct_size(x))
        {
            return;
        };

        // Allocate them (amplitude specific)
        allocate_parameters(x);
    };

    // Momentum of the photon
    double raw_inclusive_process::qGamma()
    {
        return (_s - _mT2) / sqrt(4*_s);
    };

    // Max momentum of the produced particle
    double raw_inclusive_process::pMax()
    {
        return sqrt(Kallen(_s, _mX2, minimum_M2())) / (2*sqrt(_s));
    };

    // ---------------------------------------------------------------------------
    // POLAR VARIABLES (r, cos)
    
    // Energy of the particle X
    double raw_inclusive_process::EfromRCOS(double r, double cos)
    {
        return sqrt(_mX2 + pMax()*pMax()*r*r);
    };

    // Jacobian in polar coordinates
    double raw_inclusive_process::jacobianRCOS(double r, double cos)
    {
        return(2*PI*r*r*pow(pMax(), 3.)) /  EfromRCOS(r, cos);
    };

    // ---------------------------------------------------------------------------
    // CARTESIAN VARIABLES (x, y)

    // Energy of X in (x, y)
    double raw_inclusive_process::EfromXY(double x, double y)
    {
        return sqrt(_mX2 + pMax()*pMax() * (x*x + y*y));
    };

    // Jacobian in (x, y)
    double raw_inclusive_process::jacobianXY(double x, double y)
    {
        return (2.* M_PI * y * pow(pMax(), 3)) / EfromXY(x, y);
    };

    // Energy of X in (x, y2)
    double raw_inclusive_process::EfromXY2(double x, double y2)
    {
        return sqrt(_mX2 + pMax()*pMax() * (x*x + y2));
    };

    // Jacobian in (x, y2)
    double raw_inclusive_process::jacobianXY2(double x, double y2)
    {
        return (M_PI * pow(pMax(), 3)) / EfromXY2(x, y2);
    };

    double raw_inclusive_process::XfromRCOS(double r, double cos)
    {
        return r * cos;
    };

    double raw_inclusive_process::XfromTM2(double t, double M2)
    {
        double r   = RfromTM2(t, M2);
        double cos = COSfromTM2(t, M2);
        return XfromRCOS(r, cos);
    };

    // ---------------------------------------------------------------------------
    // INVARIANT variables (t, M2)

    // momentum of produced at a fixed missing mass
    double raw_inclusive_process::pXfromM2(double M2)
    {
        return sqrt(Kallen(_s, _mX2, M2)) / (2*sqrt(_s));
    };  

    double raw_inclusive_process::COSfromTM2(double t, double M2)
    {
        double u = _mT2 + _mX2 + M2 - _s - t;
        double num  = _s * (t - u) - _mT2*(_mX2 - M2);

        return num / (4*_s*qGamma()*pXfromM2(M2));
    };

    double raw_inclusive_process::RfromTM2(double t, double M2)
    {
        return pXfromM2(M2) / pMax();
    };

    // Jacobian in (t, M2)
    double raw_inclusive_process::jacobianTM2(double t, double M2)
    {
        return PI/ (2*sqrt(_s)*qGamma());
    };

    // t from cos and M2 
    double raw_inclusive_process::TfromM2COS(double M2, double cos)
    {
        return _mX2 - (_s - _mT2) * (_s - M2 + _mX2) / (2*_s) + 2*qGamma()*pXfromM2(M2)*cos;
    };

    double raw_inclusive_process::M2fromRCOS(double r, double cos)
    {
        return _mX2 + _s - 2*sqrt(_mX2*_s + pMax()*pMax()*r*r*_s);
    };

    double raw_inclusive_process::TfromRCOS(double r, double cos)
    {
        double M2 = M2fromRCOS(r, cos);
        return _mX2 - (_s - _mT2) * (_s - M2 + _mX2) / (2*_s) + 2*qGamma()*pMax()*r*cos;
    };

    // Bounds of integration in t at fixed M2
    double raw_inclusive_process::TMINfromM2(double M2) { return TfromM2COS(M2,  1); };
    double raw_inclusive_process::TMAXfromM2(double M2) { return TfromM2COS(M2, -1); };

    // Bounds of integration in M2 at fixed t
    double raw_inclusive_process::M2MINfromT(double t) { return minimum_M2(); };
    double raw_inclusive_process::M2MAXfromT(double t) 
    {
        return (_mT2 + _mX2 - _s - t) * (_mT2 * _mX2 - _s * t) / (_mT2 - _s) / (_mX2 - t);
    };

    // ---------------------------------------------------------------------------
    // MIXED variables (t, x)

    // Missing mass from the cartesian XY
    double raw_inclusive_process::M2fromXY(double x, double y)
    {
        return _s + _mX2 - 2*sqrt(_s)*EfromXY(x, y);
    }; 
    double raw_inclusive_process::M2fromXY2(double x, double y2)
    {
        return _s + _mX2 - 2*sqrt(_s)*EfromXY(x, sqrt(y2));
    }; 


    // Similarly, momentum transfer from XY
    double raw_inclusive_process::TfromXY(double x, double y)
    {
        return _mX2 - 2*qGamma()*(EfromXY(x, y) - pMax()*x);
    };
    double raw_inclusive_process::TfromXY2(double x, double y2){ return TfromXY(x, sqrt(y2)); };

    // Bounds of integration for T at fixed X
    double raw_inclusive_process::TMINfromX(double x) { return TfromXY(x, sqrt(1 - x*x)); };
    double raw_inclusive_process::TMAXfromX(double x) { return TfromXY(x, 0); };

    // Jacobian for mixed variables
    double raw_inclusive_process::jacobianTX(double t, double x)
    {
        double inv = qGamma() / M_PI / pMax();
        return 1./inv;
    };

    // Also useful is M2 as a function of X and T
    double raw_inclusive_process::M2fromTX(double t, double x)
    {
        double lami = Kallen(_s, 0., _mT2);
        double lamf = Kallen(_s, _mX2, minimum_M2());
        double num = _mT2 * _mX2 + _mT2 * _s + _mX2 * _s - _s*_s - 2*_s*t + sqrt(lami*lamf)*x;
        return num / (_mT2 - _s);
    };

    // ---------------------------------------------------------------------------
    // Differential cross-sections
    // OUTPUT IN NANOBARN

    double raw_inclusive_process::dsigma_dtdM2(double s, double t, double M2)
    {
        set_total_energy(s);

        double mx = (use_TX()) ? XfromTM2(t, M2) : M2; 
        return jacobianTM2(t, M2) * invariant_xsection(s, t, mx);
    };

    double raw_inclusive_process::dsigma_dtdx(double s, double t, double x)
    {
        set_total_energy(s);


        double mx = (use_TX()) ? x : M2fromTX(t,x); 
        return jacobianTX(t, x) * invariant_xsection(s, t, mx);
    };

    double raw_inclusive_process::dsigma_dxdy2(double s, double x, double y2)
    {
        set_total_energy(s);

        double t  = TfromXY2(x ,y2);
        double mx = (use_TX()) ? x : M2fromXY2(t, x); 
        return jacobianXY2(x, y2) * invariant_xsection(s, t, mx);
    }

    // ---------------------------------------------------------------------------
    // Singly-differential cross-sections in terms of (t, M2)

    double raw_inclusive_process::dsigma_dt(double s, double t)
    {
        set_total_energy(s);

        auto dSigma = [&](double M2)
        {
            return dsigma_dtdM2(s, t, M2);
        };
        ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE, ROOT::Math::Integration::kGAUSS15);
        ROOT::Math::Functor1D wF(dSigma);
        ig.SetFunction(wF);
        return ig.Integral(M2MINfromT(t), M2MAXfromT(t));
    };

    double raw_inclusive_process::dsigma_dM2(double s, double M2)
    {
        set_total_energy(s);

        auto dSigma = [&](double t)
        {
            return dsigma_dtdM2(s, t, M2);
        };
        ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE, ROOT::Math::Integration::kGAUSS15);
        ROOT::Math::Functor1D wF(dSigma);
        ig.SetFunction(wF);

        return ig.Integral(TMAXfromM2(M2), TMINfromM2(M2));
    };

    // ---------------------------------------------------------------------------
    // Singly-differential cross-sections with respect to fixed (x, y2)

    double raw_inclusive_process::dsigma_dy2(double s, double y2)
    {
        set_total_energy(s);

        auto dSigma = [&](double x)
        {
            return dsigma_dxdy2(s, x, y2);
        };
        ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE, ROOT::Math::Integration::kGAUSS15);
        ROOT::Math::Functor1D wF(dSigma);
        ig.SetFunction(wF);

        return ig.Integral(0, sqrt(1 - y2));
    };

    double raw_inclusive_process::dsigma_dx(double s, double x)
    {
        set_total_energy(s);

        auto dSigma = [&](double y2)
        {
            return dsigma_dxdy2(s, x, y2);
        };
        ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE, ROOT::Math::Integration::kGAUSS15);
        ROOT::Math::Functor1D wF(dSigma);
        ig.SetFunction(wF);

        return ig.Integral(0, 1 - x*x);
    };

    // ---------------------------------------------------------------------------
    // OUTPUT IN NANOBARN
    double raw_inclusive_process::integrated_xsection(double s)
    {
        if (sqrt(s) < sqrt(_mX2) + sqrt(minimum_M2())) return 0;
        set_total_energy(s);

        auto dSigma = [&](const double * rcos)
        {
            double r = rcos[0], cos = rcos[1];

            double t  = TfromRCOS( r, cos);
            double mx = (use_TX()) ? XfromRCOS(r, cos) : M2fromRCOS(r, cos);

            // print(s, r, cos, t, M2);
            return jacobianRCOS(r, cos) * invariant_xsection(s, t, mx); 
        };
        ROOT::Math::IntegratorMultiDim ig(ROOT::Math::IntegrationMultiDim::Type::kVEGAS, 1E-6, 1E-6);

        double min[2] = {0., -1.}, max[2] = {1., 1.};
        return ig.Integral(dSigma, 2, min, max);
    };
};