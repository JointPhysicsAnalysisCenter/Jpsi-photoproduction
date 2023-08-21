// Arrays containing helicity combinations for indexing.
// Moved to a seperate file to not clutter up reaction_kinematics.hpp
//
// Author:       Daniel Winney (2020)
// Affiliation:  Joint Physics Analysis Center (JPAC)
// Email:        dwinney@iu.edu
// ---------------------------------------------------------------------------

#ifndef HELICITIES_HPP
#define HELICITIES_HPP

#include <algorithm>
#include <iostream>
#include <vector>
#include <array>

namespace jpacPhoto
{
    // Each amplitude needs to be able to tell which frame its helicities are defined in
    enum helicity_frame{ HELICITY_ERROR, HELICITY_INDEPENDENT, S_CHANNEL, T_CHANNEL, U_CHANNEL };

    // ---------------------------------------------------------------------------
    // Massless helicity combinations
    const std::vector< std::array<int, 4> > SPIN_ZERO_HELICITIES =
    {
    //  {  γ,  p,  S,  p'}
        {  1, -1,  0, -1}, // 0
        {  1, -1,  0,  1}, // 1
        {  1,  1,  0, -1}, // 2
        {  1,  1,  0,  1}, // 3
        { -1, -1,  0, -1}, // 4
        { -1, -1,  0,  1}, // 5
        { -1,  1,  0, -1}, // 6
        { -1,  1,  0,  1}  // 7
    };

    // POS / NEG refer to ordering whether hel[0] is +-1 everything else equal
    const std::vector<int> SPIN_ZERO_POS_ITERS = {0, 1, 2, 3};
    const std::vector<int> SPIN_ZERO_NEG_ITERS = {4, 5, 6, 7};

    const std::vector< std::array<int, 4> > SPIN_ONE_HELICITIES =
    {
    //  {  γ,  p,  V,  p'}
        {  1, -1,  1, -1}, // 0
        {  1, -1,  1,  1}, // 1
        {  1, -1,  0, -1}, // 2
        {  1, -1,  0,  1}, // 3
        {  1, -1, -1, -1}, // 4
        {  1, -1, -1,  1}, // 5
        {  1,  1,  1, -1}, // 6
        {  1,  1,  1,  1}, // 7
        {  1,  1,  0, -1}, // 8
        {  1,  1,  0,  1}, // 9
        {  1,  1, -1, -1}, // 10
        {  1,  1, -1,  1}, // 11
        { -1, -1,  1, -1}, // 12
        { -1, -1,  1,  1}, // 13
        { -1, -1,  0, -1}, // 14
        { -1, -1,  0,  1}, // 15
        { -1, -1, -1, -1}, // 16
        { -1, -1, -1,  1}, // 17
        { -1,  1,  1, -1}, // 18
        { -1,  1,  1,  1}, // 19
        { -1,  1,  0, -1}, // 20
        { -1,  1,  0,  1}, // 21
        { -1,  1, -1, -1}, // 22
        { -1,  1, -1,  1}  // 23
    };

    // Correspond to amplitudes with hel[2] = +1
    // POS / NEG refer to ordering whether hel[0] is +-1 everything else equal
    const std::vector<int> SPIN_ONE_POS_ITERS = {0, 1, 6, 7, 12, 13, 18, 19};
    const std::vector<int> SPIN_ONE_NEG_ITERS = {12, 13, 18, 19, 0, 1, 6, 7};

    const std::vector< std::array<int, 4> > SPIN_TWO_HELICITIES =
    {
    //  {  γ,  p,  V,  p'}
        {  1, -1,  2, -1}, //  0
        {  1, -1,  2,  1}, //  1
        {  1, -1,  1, -1}, //  2
        {  1, -1,  1,  1}, //  3
        {  1, -1,  0, -1}, //  4
        {  1, -1,  0,  1}, //  5
        {  1, -1, -1, -1}, //  6
        {  1, -1, -1,  1}, //  7
        {  1, -1, -2, -1}, //  8
        {  1, -1, -2,  1}, //  9
        {  1,  1,  2, -1}, // 10
        {  1,  1,  2,  1}, // 11
        {  1,  1,  1, -1}, // 12
        {  1,  1,  1,  1}, // 13
        {  1,  1,  0, -1}, // 14
        {  1,  1,  0,  1}, // 15
        {  1,  1, -1, -1}, // 16
        {  1,  1, -1,  1}, // 17
        {  1,  1, -2, -1}, // 18
        {  1,  1, -2,  1}, // 19
        { -1, -1,  2, -1}, // 20
        { -1, -1,  2,  1}, // 21
        { -1, -1,  1, -1}, // 22
        { -1, -1,  1,  1}, // 23
        { -1, -1,  0, -1}, // 24
        { -1, -1,  0,  1}, // 25
        { -1, -1, -1, -1}, // 26
        { -1, -1, -1,  1}, // 27
        { -1, -1, -2, -1}, // 28
        { -1, -1, -2,  1}, // 29
        { -1,  1,  2, -1}, // 30
        { -1,  1,  2,  1}, // 31
        { -1,  1,  1, -1}, // 32
        { -1,  1,  1,  1}, // 33
        { -1,  1,  0, -1}, // 34
        { -1,  1,  0,  1}, // 35
        { -1,  1, -1, -1}, // 36
        { -1,  1, -1,  1}, // 37
        { -1,  1, -2, -1}, // 38
        { -1,  1, -2,  1}  // 39
    };

    // Correspond to amplitudes with hel[2] = +1
    // POS / NEG refer to ordering whether hel[0] is +-1 everything else equal
    const std::vector<int> SPIN_TWO_POS_ITERS = {2, 3, 12, 13, 22, 23, 32, 33};
    const std::vector<int> SPIN_TWO_NEG_ITERS = {22, 23, 32, 33, 2, 3, 12, 13};

    
    // ---------------------------------------------------------------------------
    // Massless beam with Spin-3/2 recoil 
    const std::vector< std::array<int, 4> > SPIN_ZERO_THREEHALF =
    {
    //  {  γ,  p,  S,  p'}
        {  1, -1,  0, -3}, // 0
        {  1, -1,  0, -1}, // 1
        {  1, -1,  0,  1}, // 2
        {  1, -1,  0,  3}, // 3
        {  1,  1,  0, -3}, // 4
        {  1,  1,  0, -1}, // 5
        {  1,  1,  0,  1}, // 6
        {  1,  1,  0,  3}, // 7
        { -1, -1,  0, -3}, // 8
        { -1, -1,  0, -1}, // 9
        { -1, -1,  0,  1}, // 10
        { -1, -1,  0,  3}, // 11
        { -1,  1,  0, -3}, // 12
        { -1,  1,  0, -1}, // 13
        { -1,  1,  0,  1}, // 14
        { -1,  1,  0,  3}, // 15
    };

    // POS / NEG refer to ordering whether hel[0] is +-1 everything else equal
    const std::vector<int> THREE_ZERO_POS_ITERS = {0, 1, 2,  3,  4,  5,  6,  7};
    const std::vector<int> THREE_ZERO_NEG_ITERS = {8, 9, 10, 11, 12, 13, 14, 15};

    const std::vector< std::array<int, 4> > SPIN_ONE_THREEHALF =
    {
    //  {  γ,  p,  V,  p'}
        {  1, -1,  1, -3}, // 0
        {  1, -1,  1, -1}, // 1
        {  1, -1,  1,  1}, // 2
        {  1, -1,  1,  3}, // 3
        {  1, -1,  0, -3}, // 4
        {  1, -1,  0, -1}, // 5
        {  1, -1,  0,  1}, // 6
        {  1, -1,  0,  3}, // 7
        {  1, -1, -1, -3}, // 8
        {  1, -1, -1, -1}, // 9
        {  1, -1, -1,  1}, // 10
        {  1, -1, -1,  3}, // 11
        {  1,  1,  1, -3}, // 12
        {  1,  1,  1, -1}, // 13
        {  1,  1,  1,  1}, // 14
        {  1,  1,  1,  3}, // 15
        {  1,  1,  0, -3}, // 16
        {  1,  1,  0, -1}, // 17
        {  1,  1,  0,  1}, // 18
        {  1,  1,  0,  3}, // 19
        {  1,  1, -1, -3}, // 20
        {  1,  1, -1, -1}, // 21
        {  1,  1, -1,  1}, // 22
        {  1,  1, -1,  3}, // 23
        { -1, -1,  1, -3}, // 24
        { -1, -1,  1, -1}, // 25
        { -1, -1,  1,  1}, // 26
        { -1, -1,  1,  3}, // 27
        { -1, -1,  0, -3}, // 28
        { -1, -1,  0, -1}, // 29
        { -1, -1,  0,  1}, // 30
        { -1, -1,  0,  3}, // 31
        { -1, -1, -1, -3}, // 32
        { -1, -1, -1, -1}, // 33
        { -1, -1, -1,  1}, // 34
        { -1, -1, -1,  3}, // 35
        { -1,  1,  1, -3}, // 36
        { -1,  1,  1, -1}, // 37
        { -1,  1,  1,  1}, // 38
        { -1,  1,  1,  3}, // 39
        { -1,  1,  0, -3}, // 40
        { -1,  1,  0, -1}, // 41
        { -1,  1,  0,  1}, // 42
        { -1,  1,  0,  3}, // 43
        { -1,  1, -1, -3}, // 44
        { -1,  1, -1, -1}, // 45
        { -1,  1, -1,  1}, // 46
        { -1,  1, -1,  3}, // 47
    };

    // ---------------------------------------------------------------------------
    // If we allow massive "beam"
    const std::vector< std::array<int, 4> > MASSIVE_SPIN_ZERO_HELICITIES =
    {
    //  {  γ,  p,  S,  p'}
        {  1, -1,  0, -1}, // 0
        {  1, -1,  0,  1}, // 1
        {  1,  1,  0, -1}, // 2 
        {  1,  1,  0,  1}, // 3
        {  0, -1,  0, -1}, // 4
        {  0, -1,  0,  1}, // 5 
        {  0,  1,  0, -1}, // 6
        {  0,  1,  0,  1}, // 7
        { -1, -1,  0, -1}, // 8
        { -1, -1,  0,  1}, // 9
        { -1,  1,  0, -1}, // 10
        { -1,  1,  0,  1}  // 11
    };
  
    const std::vector< std::array<int, 4> > MASSIVE_SPIN_ONE_HELICITIES =
    {
    //  {  γ,  p,  V,  p'}
        {  1, -1,  1, -1}, // 0
        {  1, -1,  1,  1}, // 1
        {  1, -1,  0, -1}, // 2
        {  1, -1,  0,  1}, // 3
        {  1, -1, -1, -1}, // 4
        {  1, -1, -1,  1}, // 5
        {  1,  1,  1, -1}, // 6
        {  1,  1,  1,  1}, // 7
        {  1,  1,  0, -1}, // 8
        {  1,  1,  0,  1}, // 9
        {  1,  1, -1, -1}, // 10
        {  1,  1, -1,  1}, // 11
        {  0, -1,  1, -1}, // 12
        {  0, -1,  1,  1}, // 13
        {  0, -1,  0, -1}, // 14
        {  0, -1,  0,  1}, // 15
        {  0, -1, -1, -1}, // 16
        {  0, -1, -1,  1}, // 17
        {  0,  1,  1, -1}, // 18
        {  0,  1,  1,  1}, // 19
        {  0,  1,  0, -1}, // 20
        {  0,  1,  0,  1}, // 21
        {  0,  1, -1, -1}, // 22
        {  0,  1, -1,  1}, // 23
        { -1, -1,  1, -1}, // 24
        { -1, -1,  1,  1}, // 25
        { -1, -1,  0, -1}, // 26
        { -1, -1,  0,  1}, // 27
        { -1, -1, -1, -1}, // 28
        { -1, -1, -1,  1}, // 29
        { -1,  1,  1, -1}, // 30
        { -1,  1,  1,  1}, // 31
        { -1,  1,  0, -1}, // 32
        { -1,  1,  0,  1}, // 33
        { -1,  1, -1, -1}, // 34
        { -1,  1, -1,  1}  // 35
    };

    inline std::vector<std::array<int, 4>> get_helicities(int mJ, int bJ, bool is_massless = true)
    {
        int mjbj = 100 * is_massless + 10 * mJ + bJ;

        switch (mjbj)
        {
            case (121): return SPIN_TWO_HELICITIES;
            case (111): return SPIN_ONE_HELICITIES;
            case (113): return SPIN_ONE_THREEHALF;
            case (101): return SPIN_ZERO_HELICITIES;
            case (103): return SPIN_ZERO_THREEHALF;
            case ( 11): return MASSIVE_SPIN_ONE_HELICITIES;
            case (  1): return MASSIVE_SPIN_ZERO_HELICITIES;
            default: 
            {
                return error("get_helicities()", 
                             "Cannot find helicities for meson spin-" + std::to_string(mJ) 
                                                + " and baryon spin-" + std::to_string(bJ) + "/2.", 
                             std::vector<std::array<int, 4>>{}); 
            }
        }
    };

    inline std::array<std::vector<int>, 2> get_iters(int mJ, int bJ, bool massless = true)
    {
        int mjbj = 100 * !massless + 10 * mJ + bJ;
        switch (mjbj)
        {
            case (  1): return {SPIN_ZERO_POS_ITERS, SPIN_ZERO_NEG_ITERS};
            case ( 11): return {SPIN_ONE_POS_ITERS,  SPIN_ONE_NEG_ITERS};
            case ( 21): return {SPIN_TWO_POS_ITERS,  SPIN_TWO_NEG_ITERS};
            case (  3): return {THREE_ZERO_POS_ITERS, THREE_ZERO_POS_ITERS};
            default:
            {
                return error("get_helicities()", 
                             "Cannot find helicities for meson spin-" + std::to_string(mJ) 
                                                + " and baryon spin-" + std::to_string(bJ) + "/2.",
                             std::array<std::vector<int>,2>{}); 
            }
        };
    };

    // Output a string of a given helicity set in format e.g. {+,+,+,+}
    // TODO: add support for spin 2 or spin 3/2 
    inline std::string print_helicities(std::array<int,4> lam)
    {
        std::array<std::string,4> lams;
        for (int i = 0; i < 4; i++)
        {
            switch (lam[i])
            {
                case  1: lams[i] = "#plus";    break;
                case  0: lams[i] = "0";        break;
                case -1: lams[i] = "#minus";   break;
                default: continue;
            };
        };
        std::string hels = "{" + lams[0] + "," + lams[1] + "," + lams[2] + "," + lams[3] + "}";
        return hels;  
    };

    inline int find_helicity(std::array<int, 4> helicities, int mj, int bj, bool is_massless = true)
    {
        std::vector<std::array<int,4>> hels = get_helicities(mj, bj, is_massless);

        auto iterator = std::find(hels.begin(), hels.end(), helicities);
        
        if (iterator != hels.end())
        {
            return iterator - hels.begin();
        }
        
        return error("find_helicity", "Cannot find helicities: " + print_helicities(helicities) + "!", -1);
    };
};

#endif