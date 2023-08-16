// Plots the total inclusive cross-sections for the Z mesons 
// Reproduces Fig. 8 of [1]
//
// OUTPUT: Z_totals.pdf
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

#include "constants.hpp"
#include "kinematics.hpp"
#include "plotter.hpp"
#include "inclusive_process.hpp"
#include "inclusive/pion_exchange.hpp"
#include "inclusive/phase_space.hpp"
#include "analytic/pseudoscalar_exchange.hpp"

void Z_totals()
{
    using namespace jpacPhoto;
    using complex = std::complex<double>;
    using pion_exchange = fixed_spin::pion_exchange;

    plotter plotter;

    // ---------------------------------------------------------------------------
    // Couplings and constants
    // ---------------------------------------------------------------------------

    // Bottom vertex coupling (pi - nucleon - nucleon)
    double g_piNN = sqrt(2) * sqrt(4*PI*13.81); 

    // Cutoff for exponential form factor
    double lambda_pi = .900;  // MeV 

    // Zc(3900) couplings 
    double gc_jpsi  = 1.91; // psi coupling before VMD scaling
    double gc_gamma = E * F_JPSI * gc_jpsi / M_JPSI;

    // Zb(10610) couplings
    double gb_upsilon1S = 0.49, gb_upsilon2S = 3.30, gb_upsilon3S = 9.22;
    double gb_gamma = E * (F_UPSILON1S * gb_upsilon1S / M_UPSILON1S 
                         + F_UPSILON2S * gb_upsilon2S / M_UPSILON2S
                         + F_UPSILON3S * gb_upsilon3S / M_UPSILON3S);  

    // Zb'(10650) couplings
    double gbp_upsilon1S = 0.21, gbp_upsilon2S = 1.47, gbp_upsilon3S = 4.8;
    double gbp_gamma = E * (F_UPSILON1S * gbp_upsilon1S / M_UPSILON1S 
                         +  F_UPSILON2S * gbp_upsilon2S / M_UPSILON2S
                         +  F_UPSILON3S * gbp_upsilon3S / M_UPSILON3S);  

    // ---------------------------------------------------------------------------
    // Z minus production
    // ---------------------------------------------------------------------------

    // Zc(3900)-
    inclusive_process Zcm = new_inclusive_process<pion_exchange>(M_ZC3900, -1, "#it{Z}_{c}(3900)^{#minus}");
    Zcm->set_parameters({gc_gamma, lambda_pi});

    // Zb(10610)-
    inclusive_process Zbm = new_inclusive_process<pion_exchange>(M_ZB10610, -1, "#it{Z}_{b}(10610)^{#minus}");
    Zbm->set_parameters({gb_gamma, lambda_pi});

    // Zb'(10650)-
    inclusive_process Zbpm = new_inclusive_process<pion_exchange>(M_ZB10650, -1, "#it{Z}_{b}(10650)^{#minus}");
    Zbpm->set_parameters({gbp_gamma, lambda_pi});

    // ---------------------------------------------------------------------------
    // Z plus production
    // ---------------------------------------------------------------------------

    // Zc(3900)-
    inclusive_process Zcp = new_inclusive_process<pion_exchange>(M_ZC3900, +1, "#it{Z}_{c}(3900)^{#plus}");
    Zcp->set_parameters({gc_gamma, lambda_pi});

    // Zb(10610)-
    inclusive_process Zbp = new_inclusive_process<pion_exchange>(M_ZB10610, +1, "#it{Z}_{b}(10610)^{#plus}");
    Zbp->set_parameters({gb_gamma, lambda_pi});

    // Zb'(10650)-
    inclusive_process Zbpp = new_inclusive_process<pion_exchange>(M_ZB10650, +1, "#it{Z}_{b}(10650)^{#plus}");
    Zbpp->set_parameters({gbp_gamma, lambda_pi});

    // ---------------------------------------------------------------------------
    // Exclusive reactions 
    // ---------------------------------------------------------------------------

    kinematics kZc  = new_kinematics( M_ZC3900);
    kinematics kZb  = new_kinematics(M_ZB10610);
    kinematics kZbp = new_kinematics(M_ZB10650);

    kZc->set_meson_JP( AXIALVECTOR);
    kZb->set_meson_JP( AXIALVECTOR);
    kZbp->set_meson_JP(AXIALVECTOR);

    // Zc(3900)
    amplitude   Zce = new_amplitude<analytic::pseudoscalar_exchange>(kZc,   M_PION, "#it{Z}_{c}(3900)");
    Zce->set_parameters({gc_gamma, g_piNN, lambda_pi});

    // Zb(10610)
    amplitude   Zbe = new_amplitude<analytic::pseudoscalar_exchange>(kZb,   M_PION, "#it{Z}_{b}(10610)");
    Zbe->set_parameters({gb_gamma, g_piNN, lambda_pi});

    // Zb'(10650)
    amplitude   Zbpe = new_amplitude<analytic::pseudoscalar_exchange>(kZbp, M_PION, "#it{Z}_{b}'(10650)");
    Zbpe->set_parameters({gbp_gamma, g_piNN, lambda_pi});

    // ---------------------------------------------------------------------------
    // Make plot
    // ---------------------------------------------------------------------------

    inclusive_process to_plot;
    amplitude     to_plot_exc;

    int N = 50;
    std::array<double,2> bounds = {5, 20};

    // Z minus plot
    auto func_m = [&](double w)
    {
        double xsec =  to_plot->integrated_xsection(w*w);
        print(w, xsec);
        return xsec;
    };

    plot m = plotter.new_plot();
    m.set_curve_points(N);
    m.set_legend(0.5, 0.7);
    m.set_logscale(false, true);
    m.set_ranges({5,20}, {2E-1, 1E2});
    m.set_labels("#it{W}_{#gammap}  [GeV]", "#sigma [nb]");

    to_plot = Zcm;
    m.add_curve( bounds, func_m, to_plot->id());
    to_plot->set_option( pion_exchange::kPwave );
    m.add_dashed(bounds, func_m);
    line();

    to_plot = Zbm;
    m.add_curve(bounds, func_m, to_plot->id());
    to_plot->set_option( pion_exchange::kPwave );
    m.add_dashed(bounds, func_m);
    line();

    to_plot = Zbpm;
    m.add_curve(bounds, func_m, to_plot->id());
    to_plot->set_option( pion_exchange::kPwave );
    line();

    // Z plus plot
    auto func_p = [&](double w)
    {
        return to_plot->integrated_xsection(w*w) + to_plot_exc->integrated_xsection(w*w);
    };

    plot p = plotter.new_plot();
    p.set_curve_points(N);
    p.set_legend(0.5, 0.7);
    p.set_logscale(false, true);
    p.set_ranges({4.9,20}, {2E-1, 1E2});
    p.set_labels("#it{W}_{#gammap}  [GeV]", "#sigma [nb]");

    to_plot = Zcp; to_plot_exc = Zce;
    p.add_curve( bounds, func_p, to_plot->id());
    p.add_dashed(sigma_w, to_plot_exc, bounds);
    line();

    to_plot = Zbp; to_plot_exc = Zbe;
    p.add_curve(bounds, func_p, to_plot->id());
    p.add_dashed(sigma_w, to_plot_exc, bounds);
    line();

    to_plot = Zbpp; to_plot_exc = Zbpe;
    p.add_curve(bounds, func_p, to_plot->id());
    p.add_dashed(sigma_w, to_plot_exc, bounds);
    line();

    // Combine plots together
    plotter.combine({2,1}, {m, p}, "Z_totals.pdf");
};