// Testing suite for lorentz_tensor class, make sure nothing is messed up
//
// ------------------------------------------------------------------------------
// Author:       Daniel Winney (2023)
// Affiliation:  Joint Physics Analysis Center (JPAC),
//               South China Normal Univeristy (SCNU)
// Email:        daniel.winney@iu.alumni.edu
//               dwinney@scnu.edu.cn
// ------------------------------------------------------------------------------

#include "constants.hpp"
#include "bilinear.hpp"
#include "lorentz_tensor.hpp"

void lorentz_test()
{
    using namespace jpacPhoto;
    using complex = std::complex<double>;

    // // only complex 

    lorentz_tensor<complex, 1> q    = lorentz_vector({0, 1, 2, 3});
    lorentz_tensor<complex, 2> qq   = tensor_product(q, q);
    lorentz_tensor<complex, 3> qqq  = tensor_product<complex,3>({q, q, q});

    auto t = LORENTZ_INDICES[0];
    auto x = LORENTZ_INDICES[1];
    auto y = LORENTZ_INDICES[2];
    auto z = LORENTZ_INDICES[3];
    
    std::vector<bool> results;

    // --------------------------------------------------------------------------
    // TENSOR PRODUCT TEST
    // --------------------------------------------------------------------------

    divider();
    print("0 - Prints the product qq = q x q");
    print("Should obey qw(i,j) = i*j");

    divider(3);
    print("mu", "nu", "qq");
    divider(3);
    bool x0 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        for (auto nu : LORENTZ_INDICES)
        {
            print(mu, nu, qq(mu, nu));
            if ( !are_equal(qq(mu,nu), +mu*+nu) ) x0 = false;
        }
    }
    line(); divider(); line();
    results.push_back(x0);

    // --------------------------------------------------------------------------
    // MATRIX CAPTURE TEST
    // --------------------------------------------------------------------------

    lorentz_tensor<dirac_matrix,1> gamq =    gamma_0()*q;
    lorentz_tensor<dirac_matrix,2> gamg = -2*gamma_0()*metric_tensor();

    divider();
    print("1 - Prints gamq = gamma_0 x q(mu) and gamg = -2 gamma_0 x g(mu,nu)");
    print("Print the elements, should just equal 2*gamma_0");

    divider(4);
    print("a", "b", "gamq(2)", "gamg(1,1)");
    divider(4);
    bool x1 = true;
    for (auto a : DIRAC_INDICES)
    {   
        for (auto b : DIRAC_INDICES)
        {
            print(a, b, gamq(y)(a,b), gamg(x,x)(a,b));
            if ( !are_equal(gamq(y)(a,b), gamg(x,x)(a,b)) || !are_equal(gamq(y)(a,b), 2*gamma_0()(a,b))) x1 = false;
        }
    }
    line(); divider(); line();
    results.push_back(x1);

    // --------------------------------------------------------------------------
    // SPINOR CAPTURE TEST
    // --------------------------------------------------------------------------

    lorentz_tensor<dirac_spinor,2> diqq =     identity<dirac_spinor>() * qq;
    lorentz_tensor<dirac_spinor,2> mdig  = -2*identity<dirac_spinor>() * metric_tensor();

    divider();
    print("2 - Prints diqq = I(a) x qq(mu,nu) and mdig = -I(a) x g(mu,nu)");
    print("Print the elements, should just equal 2*I(a)");

    divider(3);
    print("a", "diqq(1,2)", "mdig(1,1)");
    divider(3);
    bool x2 = true;
    for (auto a : DIRAC_INDICES)
    {
        print(a, diqq(x,y)(a), mdig(x,x)(a));
        if ( !are_equal(diqq(x,y)(a), mdig(x,x)(a)) || !are_equal(diqq(x,y)(a), 2)) x2 = false;
    }
    line(); divider(); line();
    results.push_back(x2);

    // --------------------------------------------------------------------------
    // SCALR BY CONSTANTS OF SAME TYPE 
    // --------------------------------------------------------------------------

    lorentz_tensor<complex,1>      mq_c = complex(-1,0)*q;
    lorentz_tensor<dirac_spinor,1> mq_s = dirac_spinor({-1,-1,-1,-1})*(identity<dirac_spinor>()*q);
    lorentz_tensor<dirac_matrix,1> mq_m = (-1*identity<dirac_matrix>())*q;
    
    divider();
    print("3 - Prints mq = x*q using x = -1 but is x & q are the same type (complex, spinor, matrix)");

    divider(4);
    print("a", "mq_c", "mq_s", "mq_m");
    divider(4);
    bool x3 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        print(mu, mq_c(mu), mq_s(mu)(DIRAC_INDICES[0]), mq_m(mu)(DIRAC_INDICES[0],DIRAC_INDICES[0]));
        if ( !are_equal(mq_c(mu), mq_s(mu)(DIRAC_INDICES[0])) || !are_equal(mq_s(mu)(DIRAC_INDICES[0]), mq_m(mu)(DIRAC_INDICES[0],DIRAC_INDICES[0]))) x3 = false;
    };
    line(); divider(); line();
    results.push_back(x3);
    
    // --------------------------------------------------------------------------
    // SCALE BY CONSTANTS OF INTEGRAL TYPE
    // --------------------------------------------------------------------------

    lorentz_tensor<complex,1>      mq_ic = -2*q;
    lorentz_tensor<dirac_spinor,1> mq_is = -2*(identity<dirac_spinor>()*q);
    lorentz_tensor<dirac_matrix,1> mq_im = -2*(identity<dirac_matrix>()*q);
    
    divider();
    print("4 - Prints mq = x*q using x = -2 but is x an int");

    divider(5);
    print("a", "mq_ic", "mq_is", "mq_im");
    divider(5);
    bool x4 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        complex rc = mq_ic(mu);
        complex rs = mq_is(mu)(DIRAC_INDICES[0]);
        complex rm = mq_im(mu)(DIRAC_INDICES[0], DIRAC_INDICES[0]);

        print(mu, rc, rs, rm);
        if (!are_equal(rc, rs) || !are_equal(rc, rm) || !are_equal(rc , -2.*(+mu))) x4 = false;
    };
    line(); divider(); line();
    results.push_back(x4);

    lorentz_tensor<complex,1>      mq_dc = -2.0*q;
    lorentz_tensor<dirac_spinor,1> mq_ds = -2.0*(identity<dirac_spinor>()*q);
    lorentz_tensor<dirac_matrix,1> mq_dm = -2.0*(identity<dirac_matrix>()*q);
    
    divider();
    print("5 - mq = x*q but x is a double");

    divider(5);
    print("a", "mq_dc", "mq_ds", "mq_dm");
    divider(5);
    bool x5 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        complex rc = mq_dc(mu);
        complex rs = mq_ds(mu)(DIRAC_INDICES[0]);
        complex rm = mq_dm(mu)(DIRAC_INDICES[0], DIRAC_INDICES[0]);

        print(mu, rc, rs, rm);
        if (!are_equal(rc, rs) || !are_equal(rc, rm) || !are_equal(rc , -2.*(+mu))) x5 = false;
    };
    line(); divider(); line();
    results.push_back(x5);

    lorentz_tensor<complex,1> mq_cc = complex(-2.0)*q;
    lorentz_tensor<dirac_spinor,1> mq_cs = complex(-2.0)*(identity<dirac_spinor>()*q);
    lorentz_tensor<dirac_matrix,1> mq_cm = complex(-2.0)*(identity<dirac_matrix>()*q);
    
    divider();
    print("6 - mq = x*q but x is complex");

    divider(5);
    print("a", "mq_cc", "mq_cs", "mq_cm");
    divider(5);
    bool x6 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        complex rc = mq_cc(mu);
        complex rs = mq_cs(mu)(DIRAC_INDICES[0]);
        complex rm = mq_cm(mu)(DIRAC_INDICES[0], DIRAC_INDICES[0]);

        print(mu, rc, rs, rm);
        if (!are_equal(rc, rs) || !are_equal(rc, rm) || !are_equal(rc , -2.*(+mu))) x6 = false;
    };
    line(); divider(); line();
    results.push_back(x6);

    lorentz_tensor<complex,1>      hmq_cc = q/complex(-2.);
    lorentz_tensor<dirac_spinor,1> hmq_cs = (identity<dirac_spinor>()*q)/complex(-2.);
    lorentz_tensor<dirac_matrix,1> hmq_cm = (identity<dirac_matrix>()*q)/complex(-2.);
    
    divider();
    print("7 - Calculate hmq = q / x for x = -2 is complex");

    divider(5);
    print("a", "mq_cc", "mq_cs", "mq_cm");
    divider(5);
    bool x7 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        complex rc = hmq_cc(mu);
        complex rs = hmq_cs(mu)(DIRAC_INDICES[0]);
        complex rm = hmq_cm(mu)(DIRAC_INDICES[0], DIRAC_INDICES[0]);

        print(mu, rc, rs, rm);
        if (!are_equal(rc, rs) || !are_equal(rc, rm) || !are_equal(rc , (+mu)/(-2.))) x7 = false;
    };
    line(); divider(); line();
    results.push_back(x7);

    lorentz_tensor<complex,1>      hmq_ic = q/(-2);
    lorentz_tensor<dirac_spinor,1> hmq_is = (identity<dirac_spinor>()*q)/(-2);
    lorentz_tensor<dirac_matrix,1> hmq_im = (identity<dirac_matrix>()*q)/(-2);
    
    divider();
    print("8 - Same as above except x is int");

    divider(5);
    print("a", "mq_ic", "mq_is", "mq_im");
    divider(5);
    bool x8 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        complex rc = hmq_ic(mu);
        complex rs = hmq_is(mu)(DIRAC_INDICES[0]);
        complex rm = hmq_im(mu)(DIRAC_INDICES[0], DIRAC_INDICES[0]);

        print(mu, rc, rs, rm);
        if (!are_equal(rc, rs) || !are_equal(rc, rm) || !are_equal(rc , (+mu)/(-2.))) x8 = false;
    };
    line(); divider(); line();
    results.push_back(x8);

    lorentz_tensor<complex,1>      mq_uc = -q;
    lorentz_tensor<dirac_spinor,1> mq_us = -(identity<dirac_spinor>()*q);
    lorentz_tensor<dirac_matrix,1> mq_um = -(identity<dirac_matrix>()*q);
    
    divider();
    print("9 - Unary negation operator");

    divider(5);
    print("a", "mq_uc", "mq_us", "mq_um");
    divider(5);
    bool x9 = true;
    for (auto mu : LORENTZ_INDICES)
    {
        complex rc = mq_uc(mu);
        complex rs = mq_us(mu)(DIRAC_INDICES[0]);
        complex rm = mq_um(mu)(DIRAC_INDICES[0], DIRAC_INDICES[0]);

        print(mu, rc, rs, rm);
        if (!are_equal(rc, rs) || !are_equal(rc, rm) || !are_equal(rc , -1.*(+mu))) x9 = false;
    };
    line(); divider(); line();
    results.push_back(x9);


    auto pass = [](bool x)
    {
        return x ? "pass" : "fail";
    };

    line(); line(); divider(2);
    print("Test", "Result");
    divider(2);
    for (int i = 0; i < results.size(); i++)
    {
        print(i, pass(results[i]));
    };
    divider(2); line();
};