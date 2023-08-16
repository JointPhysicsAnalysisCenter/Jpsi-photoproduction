// 
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------
// REFERENCES:
// [1] 	arXiv:2209.05882 [hep-ph]
// ------------------------------------------------------------------------------

#include "inclusive_process.hpp"
#include "inclusive/pion_exchange.hpp"
#include "inclusive/phase_space.hpp"
#include "analytic/pseudoscalar_exchange.hpp"

#include "Math/GSLIntegrator.h"
#include "Math/IntegrationTypes.h"
#include "plotter.hpp"

using namespace jpacPhoto;

void b1()
{
    using namespace jpacPhoto;
    using complex = std::complex<double>;
    using pion_exchange = fixed_spin::pion_exchange;
    
    plotter plotter;

    // ---------------------------------------------------------------------------
    // Couplings
    // ---------------------------------------------------------------------------

    double g_b1 = 0.24;
    double g_NN = sqrt(2.) * sqrt(4. * PI * 13.81); // Nucleon coupling same for all
    double g_delta = 18.5;
    double LamPi = .9;  // 900 MeV cutoff for formfactor
    
    // --------------------------------s-------------------------------------------
    // Exclusive amplitudes
    // ---------------------------------------------------------------------------

    kinematics kb1N = new_kinematics(M_B1);
    kb1N->set_meson_JP( AXIALVECTOR );

    amplitude b1N   = new_amplitude<analytic::pseudoscalar_exchange>(kb1N, M_PION, "b_{1} exclusive");
    b1N->set_parameters({g_b1, g_NN, LamPi});

    kinematics kb1D = new_kinematics(M_B1, M_DELTA);
    kb1D->set_meson_JP( AXIALVECTOR );
    kb1D->set_baryon_JP( THREEPLUS );

    amplitude b1D   = new_amplitude<analytic::pseudoscalar_exchange>(kb1D, M_PION, "b_{1} exclusive");
    b1D->set_parameters({g_b1, g_delta, LamPi});

    // --------------------------------s-------------------------------------------
    // Auxilary functions to add the delta -> pi N lineshape
    // ---------------------------------------------------------------------------

     // Sill line-shape
    // Mass and width from 2106.03749
    auto Sill = [&](double w)
    {
        double width = 120.4E-3;
        double mass  = 1236.2E-3;
        double mN = 0.938, mpi = 0.134;
        double sth = (mN + mpi)*(mN+mpi);

        double gamma = width*mass / sqrt(mass*mass - sth);

        return 2.*w/M_PI * sqrt(w*w - sth) *gamma / (pow(w*w-mass*mass, 2.) + pow(sqrt(w*w - sth)*gamma, 2.));
    };

    auto func_PiN = [&](double w)
    {
        auto dH = [&](double m)
        {
            
            kb1D->set_recoil_mass(m);
            return Sill(m)*b1D->integrated_xsection(w*w) * 1E-3; // in mub!
        };

        ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE, ROOT::Math::Integration::kGAUSS15);
        ROOT::Math::Functor1D wH(dH);
        ig.SetFunction(wH);
        
        return ig.Integral(M_PION+M_PROTON, 3.);
    };

    // ---------------------------------------------------------------------------
    // Inclusives
    // ---------------------------------------------------------------------------

    inclusive_process b1p = new_inclusive_process<pion_exchange>(M_B1, +1, "b_{1}(1235)^{#plus}");
    b1p->set_parameters({g_b1, LamPi});
    
    inclusive_process b1m = new_inclusive_process<pion_exchange>(M_B1, -1, "b_{1}(1235)^{#minus}");
    b1m->set_parameters({g_b1, LamPi});

    // ---------------------------------------------------------------------------
    // Make plot
    // ---------------------------------------------------------------------------

    int N = 30;
    std::array<double,2> bounds = {2, 4};

    // b1 minus plot
    auto func_m = [&](double w)
    {
        return b1m->integrated_xsection(w*w) * 1E-3; // in mub
    };

    // b1 plus plot
    auto func_p = [&](double w)
    {
        return (b1p->integrated_xsection(w*w) + b1N->integrated_xsection(w*w)) * 1E-3; // in mub
    };

    auto func_N = [&](double w)
    {
        return b1N->integrated_xsection(w*w) * 1E-3; // in mub
    };

    auto func_D = [&](double w)
    {
        return b1D->integrated_xsection(w*w) * 1E-3; // in mub
    };

    plot p1 = plotter.new_plot();
    p1.set_curve_points(N);
    p1.set_legend(0.5, 0.7);
    p1.set_ranges({2, 4}, {0, 10});
    p1.set_labels("#it{W}_{#gammap}  [GeV]", "#sigma [#mub]");

    p1.add_curve( bounds, func_PiN, "b_{1}^{#minus} (#Delta^{#plus#plus}#rightarrow#pi^{#plus} #it{p}) from BW");
    kb1D->set_recoil_mass(M_B1);
    p1.add_dashed( bounds, func_D);

    b1m->set_option(pion_exchange::kPwave);
    p1.add_curve( bounds, func_m, "b_{1}^{#minus} (#Delta^{#plus#plus}#rightarrow#pi^{#plus} #it{p}) from SAID");
    b1m->set_option(pion_exchange::kJPAC);
    p1.add_curve( bounds, func_m, "Inclusive b_{1}^{#minus}");

    plot p2 = plotter.new_plot();
    p2.set_curve_points(N);
    p2.set_legend(0.5, 0.7);
    p2.set_ranges({2, 5}, {0, 8});
    p2.set_labels("#it{W}_{#gammap}  [GeV]", "#sigma [#mub]");
    p2.add_curve( bounds, func_p, b1p->id());
    p2.add_dashed(bounds, func_N);
    p2.add_curve( bounds, func_m, b1m->id());

    plotter.combine({2,1}, {p1, p2}, "b1.pdf");
};