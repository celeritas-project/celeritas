//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/ChannelSelector.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "celeritas/mucf/executor/detail/DDChannelSelector.hh"
#include "celeritas/mucf/executor/detail/DTChannelSelector.hh"
#include "celeritas/mucf/executor/detail/TTChannelSelector.hh"

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

class ChannelSelectorTest : public Test
{
  protected:
    using Engine = DiagnosticRngEngine<std::mt19937>;

    void SetUp() override { rng_.reset_count(); }

    Engine& rng() { return rng_; }

    // Sticking fraction between the two dd --> 3He channels
    double dd_sticking_fraction() { return 0.122; }
    // Sticking fraction for dt
    double dt_sticking_fraction() { return 0.00857; }
    // Sticking fraction for tt
    double tt_sticking_fraction() { return 0.14; }

    // Calculate dd --> 3He channel probability from the branching ratio
    double he3_probability(double branching_ratio)
    {
        return branching_ratio / (branching_ratio + 1);
    }

    // Calculate sigma for the statistical tests
    double calc_sigma(double num_samples, double success_prob)
    {
        return std::sqrt(num_samples * success_prob * (1 - success_prob));
    }

  private:
    Engine rng_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ChannelSelectorTest, dd_channel_low_temperature)
{
    // dd fusion at T < 50 K: branching_ratio = 1
    double const temperature = 30.0;
    double const branching_ratio = 1.0;
    double const he3_probability = this->he3_probability(branching_ratio);
    double const sticking_fraction = this->dd_sticking_fraction();

    DDChannelSelector select_channel(temperature);

    int num_samples = 100000;
    int helium3_count = 0;
    int muonichelium3_count = 0;
    int tritium_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto channel = select_channel(this->rng());
        if (channel == DDMucfInteractor::Channel::helium3_muon_neutron)
        {
            helium3_count++;
        }
        else if (channel == DDMucfInteractor::Channel::muonichelium3_neutron)
        {
            muonichelium3_count++;
        }
        else if (channel == DDMucfInteractor::Channel::tritium_muon_proton)
        {
            tritium_count++;
        }
        else
        {
            FAIL() << "Unexpected channel selected";
        }
    }

    EXPECT_EQ(num_samples, helium3_count + muonichelium3_count + tritium_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        double const num_samples_d = static_cast<double>(num_samples);

        double const expected_tritium_count = num_samples_d
                                              * (1 - he3_probability);
        double const expected_helium3_count = num_samples_d * he3_probability
                                              * (1 - sticking_fraction);
        double const expected_muonichelium3_count
            = num_samples_d * he3_probability * sticking_fraction;
        double const tolerance
            = 3 * this->calc_sigma(num_samples_d, he3_probability);

        EXPECT_NEAR(expected_tritium_count, tritium_count, tolerance);
        EXPECT_NEAR(expected_helium3_count, helium3_count, tolerance);
        EXPECT_NEAR(
            expected_muonichelium3_count, muonichelium3_count, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ChannelSelectorTest, dd_channel_mid_temperature)
{
    // DD fusion at 50 < T < 100 K: branching_ratio = 1.0088 * (T - 50) = 25.22
    double const temperature = 75.0;
    double const branching_ratio = 1.0088 * (temperature - 50);
    double const he3_probability = this->he3_probability(branching_ratio);

    DDChannelSelector select_channel(temperature);

    size_type const num_samples = 10000;
    size_type he3_total_count = 0;
    size_type tritium_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto channel = select_channel(this->rng());
        if (channel == DDMucfInteractor::Channel::helium3_muon_neutron
            || channel == DDMucfInteractor::Channel::muonichelium3_neutron)
        {
            he3_total_count++;
        }
        else if (channel == DDMucfInteractor::Channel::tritium_muon_proton)
        {
            tritium_count++;
        }
    }

    EXPECT_EQ(num_samples, he3_total_count + tritium_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        double const num_samples_d = static_cast<double>(num_samples);
        double const he3_total_count_d = static_cast<double>(he3_total_count);

        double const expected_he3_count = num_samples_d * he3_probability;
        double const tolerance
            = 3 * this->calc_sigma(num_samples_d, he3_probability);

        EXPECT_NEAR(expected_he3_count, he3_total_count_d, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ChannelSelectorTest, dd_channel_high_temperature)
{
    // DD fusion at T < 300 K: branching_ratio = 1.44
    double const temperature = 300;
    double const branching_ratio = 1.44;
    double const he3_probability = this->he3_probability(branching_ratio);

    DDChannelSelector select_channel(temperature);

    size_type const num_samples = 10000;
    size_type he3_total_count = 0;
    size_type tritium_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto channel = select_channel(this->rng());
        if (channel == DDMucfInteractor::Channel::helium3_muon_neutron
            || channel == DDMucfInteractor::Channel::muonichelium3_neutron)
        {
            he3_total_count++;
        }
        else if (channel == DDMucfInteractor::Channel::tritium_muon_proton)
        {
            tritium_count++;
        }
    }

    EXPECT_EQ(num_samples, he3_total_count + tritium_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        double const num_samples_d = static_cast<double>(num_samples);
        double const he3_total_count_d = static_cast<double>(he3_total_count);

        double const expected_he3_count = num_samples_d * he3_probability;
        // 3 sigma tolerance
        double const tolerance
            = 3 * this->calc_sigma(num_samples_d, he3_probability);

        EXPECT_NEAR(expected_he3_count, he3_total_count_d, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ChannelSelectorTest, dd_sticking_fraction_within_he3)
{
    // Test that when He3 channel is selected, sticking fraction is 12.2%
    double const temperature = 300;
    double const sticking_fraction = this->dd_sticking_fraction();

    DDChannelSelector select_channel(temperature);

    size_type const num_samples = 10000;
    size_type helium3_count = 0;
    size_type muonichelium3_count = 0;

    // Only count He3 channels
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto channel = select_channel(this->rng());
        if (channel == DDMucfInteractor::Channel::helium3_muon_neutron)
        {
            helium3_count++;
        }
        else if (channel == DDMucfInteractor::Channel::muonichelium3_neutron)
        {
            muonichelium3_count++;
        }
    }

    size_type total_he3 = helium3_count + muonichelium3_count;
    EXPECT_GT(total_he3, 0);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        double const total_he3_d = static_cast<double>(total_he3);
        double const muonichelium3_count_d
            = static_cast<double>(muonichelium3_count);

        double const expected_muonichelium3 = total_he3_d * sticking_fraction;
        // 3 sigma tolerance
        double const tolerance
            = 3 * this->calc_sigma(total_he3_d, sticking_fraction);

        EXPECT_NEAR(expected_muonichelium3, muonichelium3_count_d, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ChannelSelectorTest, dt_channel)
{
    // DT fusion: ~0.8% sticking
    DTChannelSelector select_channel;

    size_type const num_samples = 100000;
    size_type alpha_count = 0;
    size_type muonicalpha_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto channel = select_channel(this->rng());
        if (channel == DTMucfInteractor::Channel::alpha_muon_neutron)
        {
            alpha_count++;
        }
        else if (channel == DTMucfInteractor::Channel::muonicalpha_neutron)
        {
            muonicalpha_count++;
        }
        else
        {
            FAIL() << "Unexpected channel selected";
        }
    }

    EXPECT_EQ(num_samples, alpha_count + muonicalpha_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        double const num_samples_d = static_cast<double>(num_samples);
        double const alpha_count_d = static_cast<double>(alpha_count);
        double const muonicalpha_count_d
            = static_cast<double>(muonicalpha_count);

        double const sticking_fraction = this->dt_sticking_fraction();
        double const expected_muonicalpha_count = num_samples_d
                                                  * sticking_fraction;
        double const expected_alpha_count = num_samples_d
                                            * (1 - sticking_fraction);
        // 3 sigma tolerance
        double const tolerance
            = 3 * this->calc_sigma(num_samples_d, sticking_fraction);

        EXPECT_NEAR(expected_muonicalpha_count, muonicalpha_count_d, tolerance);
        EXPECT_NEAR(expected_alpha_count, alpha_count_d, tolerance);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ChannelSelectorTest, tt_channel)
{
    // TT fusion: ~14% sticking
    TTChannelSelector select_channel;

    size_type const num_samples = 10000;
    size_type alpha_count = 0;
    size_type muonicalpha_count = 0;

    for ([[maybe_unused]] auto i : range(num_samples))
    {
        auto channel = select_channel(this->rng());
        if (channel == TTMucfInteractor::Channel::alpha_muon_neutron_neutron)
        {
            alpha_count++;
        }
        else if (channel
                 == TTMucfInteractor::Channel::muonicalpha_neutron_neutron)
        {
            muonicalpha_count++;
        }
        else
        {
            FAIL() << "Unexpected channel selected";
        }
    }

    EXPECT_EQ(num_samples, alpha_count + muonicalpha_count);

    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        double const num_samples_d = static_cast<double>(num_samples);
        double const alpha_count_d = static_cast<double>(alpha_count);
        double const muonicalpha_count_d
            = static_cast<double>(muonicalpha_count);

        double const sticking_fraction = this->tt_sticking_fraction();
        double const expected_muonicalpha_count = num_samples_d
                                                  * sticking_fraction;
        double const expected_alpha_count = num_samples_d
                                            * (1 - sticking_fraction);

        double const tolerance
            = 3 * this->calc_sigma(num_samples_d, sticking_fraction);

        EXPECT_NEAR(expected_muonicalpha_count, muonicalpha_count_d, tolerance);
        EXPECT_NEAR(expected_alpha_count, alpha_count_d, tolerance);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
