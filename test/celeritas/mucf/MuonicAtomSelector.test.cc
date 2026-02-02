//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/MuonicAtomSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/mucf/executor/detail/MuonicAtomSelector.hh"

#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "celeritas/mucf/executor/detail/MuonicAtomSpinSelector.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MuonicAtomSelectorTest : public Test
{
  protected:
    using Engine = DiagnosticRngEngine<std::mt19937>;

    void SetUp() override {}

    real_type
    calc_deuterium_q1s_prob(real_type deuterium_frac, real_type tritium_frac)
    {
        real_type q1s = 1.0 / (1.0 + 2.9 * tritium_frac);
        return deuterium_frac * q1s;
    }

    real_type calc_sigma(real_type num_samples, real_type success_prob)
    {
        return std::sqrt(num_samples * success_prob * (1 - success_prob));
    }

  protected:
    Engine rng_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MuonicAtomSelectorTest, muonic_atom_pure_deuterium)
{
    // Pure deuterium
    MuonicAtomSelector select_atom(1, 0);

    size_type const num_samples = 100;
    size_type deuterium_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto atom = select_atom(rng_);
        if (atom == MucfMuonicAtom::deuterium)
        {
            deuterium_count++;
        }
    }
    EXPECT_EQ(num_samples, deuterium_count);
}

//---------------------------------------------------------------------------//
TEST_F(MuonicAtomSelectorTest, muonic_atom_pure_tritium)
{
    // Pure tritium
    MuonicAtomSelector select_atom(0, 1);

    size_type const num_samples = 100;
    size_type tritium_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto atom = select_atom(rng_);
        if (atom == MucfMuonicAtom::tritium)
        {
            tritium_count++;
        }
    }
    EXPECT_EQ(num_samples, tritium_count);
}

//---------------------------------------------------------------------------//
TEST_F(MuonicAtomSelectorTest, muonic_atom_dt_mixture)
{
    // 50/50 mixture
    real_type const d_frac = 0.5;
    real_type const t_frac = 0.5;
    MuonicAtomSelector select_atom(d_frac, t_frac);

    size_type const num_samples = 10000;
    size_type deuterium_count = 0;
    size_type tritium_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto atom = select_atom(rng_);
        if (atom == MucfMuonicAtom::deuterium)
        {
            deuterium_count++;
        }
        else if (atom == MucfMuonicAtom::tritium)
        {
            tritium_count++;
        }
    }

    EXPECT_EQ(num_samples, deuterium_count + tritium_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        real_type const expected_d_prob
            = this->calc_deuterium_q1s_prob(d_frac, t_frac);
        real_type const expected_d_count = num_samples * expected_d_prob;
        // 3 sigma tolerance
        real_type const tolerance
            = 3 * this->calc_sigma(num_samples, expected_d_prob);

        EXPECT_NEAR(expected_d_count, deuterium_count, tolerance);
        EXPECT_NEAR(num_samples - expected_d_count, tritium_count, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuonicAtomSelectorTest, muonic_atom_asymmetric_mixture)
{
    // 70/30 mixture
    real_type const d_frac = 0.7;
    real_type const t_frac = 0.3;
    MuonicAtomSelector select_atom(d_frac, t_frac);

    size_type const num_samples = 10000;
    size_type deuterium_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto atom = select_atom(rng_);
        if (atom == MucfMuonicAtom::deuterium)
        {
            deuterium_count++;
        }
    }

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        real_type const expected_d_prob
            = this->calc_deuterium_q1s_prob(d_frac, t_frac);
        real_type const expected_d_count = num_samples * expected_d_prob;
        // 3 sigma tolerance
        real_type const tolerance
            = 3 * this->calc_sigma(num_samples, expected_d_prob);

        EXPECT_NEAR(expected_d_count, deuterium_count, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuonicAtomSelectorTest, spin_selector_deuterium)
{
    MuonicAtomSpinSelector select_spin(MucfMuonicAtom::deuterium);

    size_type const num_samples = 10000;
    size_type spin_3_2_count = 0;  // Spin 3/2
    size_type spin_1_2_count = 0;  // Spin 1/2

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto spin = select_spin(rng_);
        if (spin == units::HalfSpinInt{3})
        {
            spin_3_2_count++;
        }
        else if (spin == units::HalfSpinInt{1})
        {
            spin_1_2_count++;
        }
        else
        {
            FAIL() << "Unexpected spin value: " << spin.value();
        }
    }

    EXPECT_EQ(num_samples, spin_3_2_count + spin_1_2_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        // Expected: 2/3 for spin 3/2, 1/3 for spin 1/2
        real_type expected_3_2_prob = 2.0 / 3.0;
        real_type expected_3_2_count = num_samples * expected_3_2_prob;
        // 3 sigma tolerance
        real_type tolerance
            = 3 * this->calc_sigma(num_samples, expected_3_2_prob);

        EXPECT_NEAR(expected_3_2_count, spin_3_2_count, tolerance);
        EXPECT_NEAR(
            num_samples - expected_3_2_count, spin_1_2_count, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuonicAtomSelectorTest, spin_selector_tritium)
{
    MuonicAtomSpinSelector select_spin(MucfMuonicAtom::tritium);

    size_type const num_samples = 10000;
    size_type spin_1_count = 0;  // Spin 1
    size_type spin_0_count = 0;  // Spin 0

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto spin = select_spin(rng_);
        if (spin == units::HalfSpinInt{2})
        {
            spin_1_count++;
        }
        else if (spin == units::HalfSpinInt{0})
        {
            spin_0_count++;
        }
        else
        {
            FAIL() << "Unexpected spin value: " << spin.value();
        }
    }

    EXPECT_EQ(num_samples, spin_1_count + spin_0_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        // Expected: 3/4 for spin 1, 1/4 for spin 0
        real_type const expected_1_prob = 0.75;
        real_type const expected_1_count = num_samples * expected_1_prob;
        // 3 sigma tolerance
        real_type const tolerance
            = 3 * this->calc_sigma(num_samples, expected_1_prob);

        EXPECT_NEAR(expected_1_count, spin_1_count, tolerance);
        EXPECT_NEAR(num_samples - expected_1_count, spin_0_count, tolerance);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
