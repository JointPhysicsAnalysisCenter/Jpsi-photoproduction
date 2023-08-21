// Draws the Chew-Low plot (physical region) for the b1 inclusive kinematics
// Reproduces Fig. 11 of [1]
//
// OUTPUT: chew_low.pdf 
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
#include "plotter.hpp"

using namespace jpacPhoto;

void chew_low()
{
    using namespace jpacPhoto;
    using complex = std::complex<double>;

    inclusive_process b1 = new_inclusive_process<phase_space>(M_B1, M_PROTON + M_PION, "b_{1} Phase space");
    b1->set_total_energy(9);


    double W = M_PROTON + M_PION;
    std::array<double,2> bounds = {-b1->TMINfromM2(W*W) , -b1->TMAXfromM2(W*W)};

    plotter plotter;
    plot p = plotter.new_plot();
    p.set_ranges({0, 5.5}, {1, 3.3});

    auto M2max = [&](double mt)
    {
        return b1->M2MAXfromT(-mt);
    };
    auto M2min = [&](double mt)
    {
        return b1->M2MINfromT(-mt);
    };
    p.add_curve(bounds, M2max, "");
    p.add_curve(bounds, M2min, "");
    p.set_legend(false);
    p.set_labels( "#minus#it{t} [GeV^{2}]", "#it{M}^{2}  [GeV^{2}]");
    p.save("chew_low.pdf");
};