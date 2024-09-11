//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/NuclearFormFactors.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/xs/NuclearFormFactors.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class NuclearFormFactorsTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}

    template<class F>
    auto evaluate_all()
    {
        std::vector<real_type> result;
        for (auto amass : atomic_mass_numbers)
        {
            F calc_ff{AtomicNumber{amass}};
            for (auto mom : momentum)
            {
                result.push_back(calc_ff(units::MevMomentum{mom}));
            }
        }
        return result;
    }

    static int const atomic_mass_numbers[4];
    static real_type const momentum[5];
};

//! Atomic mass for common H, Be, Si, Cu isotopes
int const NuclearFormFactorsTest::atomic_mass_numbers[4] = {1, 9, 28, 63};
//! Momentum [MeV/c] corresponding to {0.05, 0.1, 0.5, 1.0, 10.0} MeV
real_type const NuclearFormFactorsTest::momentum[5] = {
    0.05359989461, 0.11219978922, 0.7609989461, 2.0219978922, 110.219978922};

TEST_F(NuclearFormFactorsTest, exponential)
{
    auto result = this->evaluate_all<ExpNuclearFormFactor>();
    // clang-format off
    static double const expected_result[] = {
        0.99999999111533, 0.99999996106881, 0.99999820905938, 0.99998735639443, 0.96346328587772,
        0.99999993503162, 0.99999971531919, 0.99998690401855, 0.99990755030365, 0.77304182886306,
        0.9999998800842 , 0.99999947454867, 0.99997582820929, 0.99982937038808, 0.63639467627637,
        0.99999981419604, 0.99999918583774, 0.99996254730628, 0.99973563593242, 0.51546164592414,
    };
    // clang-format on
    EXPECT_VEC_SOFT_EQ(expected_result, result);

    // Test prefactor value for special case and iron
    EXPECT_SOFT_EQ(1.5462640713992254e-06,
                   ExpNuclearFormFactor{AtomicNumber{1}}.prefactor().value());
    EXPECT_SOFT_EQ(3.0344136960050336e-05,
                   ExpNuclearFormFactor{AtomicNumber{56}}.prefactor().value());
}

TEST_F(NuclearFormFactorsTest, gaussian)
{
    auto result = this->evaluate_all<GaussianNuclearFormFactor>();
    // clang-format off
    static double const expected_result[] = {
        0.99999999111533, 0.99999996106881, 0.99999820905857, 0.99998735635446, 0.96312757030345,
        0.99999993503162, 0.99999971531917, 0.99998690397568, 0.99990754816688, 0.75978263411599,
        0.99999988008419, 0.9999994745486 , 0.99997582806322, 0.99982936310926, 0.60225668401895,
        0.99999981419603, 0.99999918583758, 0.9999625469556 , 0.99973561845956, 0.45580791667755,
    };
    // clang-format on
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

TEST_F(NuclearFormFactorsTest, uniform_uniform_folded)
{
    if (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE)
    {
        GTEST_SKIP() << "UU suffers from numerical errors for anything below "
                        "~2 MeV";
    }
    auto result = this->evaluate_all<UUNuclearFormFactor>();
    // clang-format off
    static double const expected_result[] = {
        0.99999995727013, 0.99999982428955, 0.99999190917794, 0.99994288145528, 0.84170249371756,
        0.99999992450445, 0.99999966921585, 0.99998478436884, 0.99989258467313, 0.72091728794807,
        0.99999987267501, 0.9999994414925 , 0.99997430281983, 0.99981859418499, 0.56584126075538,
        0.99999980235623, 0.99999913361891, 0.99996014223619, 0.99971863830384, 0.39399697953588,
    };
    // clang-format on
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
