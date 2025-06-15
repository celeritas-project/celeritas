//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/ScreeningFunctions.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/xs/ScreeningFunctions.hh"

#include "corecel/grid/VectorUtils.hh"
#include "celeritas/UnitTypes.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class ScreeningFunctionsTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(ScreeningFunctionsTest, all)
{
    using Mass = units::MevMass;
    using InvEnergy = RealQuantity<UnitInverse<units::Mev>>;

    // Element abbreviations: C, Al, Fe, W
    static int const all_z[] = {6, 13, 26, 74};
    Mass const emass{0.511};

    // Store the results for each element
    std::vector<std::vector<real_type>> phi1;
    std::vector<std::vector<real_type>> dphi;
    std::vector<std::vector<real_type>> psi1;
    std::vector<std::vector<real_type>> dpsi;

    // Generate a sequence of energies to test
    auto energies = geomspace(1e-3, 1e6, 5);

    for (auto z : all_z)
    {
        auto cbrt_z = std::cbrt(static_cast<real_type>(z));
        auto delta = emass * (100 / cbrt_z);
        auto epsilon = emass * (100 / ipow<2>(cbrt_z));

        TsaiScreeningCalculator calc_screening{delta, epsilon};

        // Initialize vectors for this element
        std::vector<real_type> el_phi1;
        std::vector<real_type> el_dphi;
        std::vector<real_type> el_psi1;
        std::vector<real_type> el_dpsi;

        for (auto e : energies)
        {
            auto f = calc_screening(InvEnergy{static_cast<real_type>(1 / e)});

            el_phi1.push_back(f.phi1);
            el_dphi.push_back(f.dphi);
            el_psi1.push_back(f.psi1);
            el_dpsi.push_back(f.dpsi);
        }
        phi1.push_back(std::move(el_phi1));
        dphi.push_back(std::move(el_dphi));
        psi1.push_back(std::move(el_psi1));
        dpsi.push_back(std::move(el_dpsi));
    }
    static std::vector<real_type> const expected_phi1[] = {
        {-21.783857353441,
         -1.0608479235529,
         17.921623316243,
         20.840250048675,
         20.862871768052},
        {-20.752937507931,
         -0.030100982565447,
         18.451891110572,
         20.845409362847,
         20.862900901554},
        {-19.828741275161,
         0.89384311509174,
         18.86045505191,
         20.849033068662,
         20.862921345345},
        {-18.43411655665,
         2.2877809333942,
         19.368224153221,
         20.853140248517,
         20.862944498335},
    };
    static std::vector<real_type> const expected_dphi[] = {
        {1.404968089095e-10,
         4.4128108010779e-06,
         0.057844334589045,
         0.64558504095719,
         0.66654482631556},
        {2.3524637671483e-10,
         7.3740645559725e-06,
         0.080313479197815,
         0.65027351743972,
         0.66657250461478},
        {3.7342550764018e-10,
         1.1678849710456e-05,
         0.10530772357891,
         0.65359847700402,
         0.66659192832806},
        {7.4994863087583e-10,
         2.3347205149048e-05,
         0.15183621017018,
         0.65739936870912,
         0.66661392713376},
    };
    static std::vector<real_type> const expected_psi1[] = {
        {-19.395134777263,
         1.3281109201117,
         21.554317398657,
         28.186171200905,
         28.339111207136},
        {-17.333295076456,
         3.3899142968228,
         23.107550539419,
         28.247122794382,
         28.339469156808},
        {-15.484902597677,
         5.2382209627956,
         24.251907855416,
         28.281137641401,
         28.339665578296},
        {-12.695653130839,
         8.0270389014542,
         25.558168359864,
         28.310539210367,
         28.339833479292},
    };
    static std::vector<real_type> const expected_dpsi[] = {
        {6.9588767322661e-12,
         2.1980777911256e-07,
         0.0057285675287362,
         0.59892214515549,
         0.66625416991133},
        {1.9510125075465e-11,
         6.1578565252681e-07,
         0.014226492156761,
         0.62490787178122,
         0.66642026688189},
        {4.9162122093685e-11,
         1.5499321186881e-06,
         0.030406389014177,
         0.63989072947245,
         0.66651142860458},
        {1.9829005302547e-10,
         6.2324110057565e-06,
         0.083031377928627,
         0.65312854821131,
         0.66658936348276},
    };

    EXPECT_VEC_NEAR(expected_phi1, phi1, 10 * fine_eps);
    EXPECT_VEC_SOFT_EQ(expected_dphi, dphi);
    EXPECT_VEC_SOFT_EQ(expected_psi1, psi1);
    EXPECT_VEC_SOFT_EQ(expected_dpsi, dpsi);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
