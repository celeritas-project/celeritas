//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/MuonicMoleculeSelector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/mucf/executor/detail/MuonicMoleculeSelector.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/random/DiagnosticRngEngine.hh"

#include "MucfInteractorHostTestBase.hh"
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

class MuonicMoleculeSelectorTest
    : public ::celeritas::test::MucfInteractorHostTestBase
{
  protected:
    using Base = ::celeritas::test::MucfInteractorHostTestBase;
    using Engine = DiagnosticRngEngine<std::mt19937>;
    using MMM = MucfMuonicMolecule;
    using MMA = MucfMuonicAtom;
    using HalfSpinInt = MuonicMoleculeSelector::HalfSpinInt;
    using CycleRatesArray = MuonicMoleculeSelector::CycleRatesArray;
    using MoleculeCountArray = EnumArray<MucfMuonicMolecule, size_type>;

    void SetUp() override
    {
        data_ = this->host_data();
        rng_.reset_count();

        auto const mucfmatid = MuCfMatId{0};
        cycle_rates_ = data_.cycle_rates[mucfmatid];
    }

    MoleculeCountArray sample_molecules(MucfMuonicAtom atom,
                                        HalfSpinInt spin,
                                        CycleRatesArray cycle_rates,
                                        size_type num_samples)
    {
        MuonicMoleculeSelector select_molecule(atom, spin, cycle_rates);

        MoleculeCountArray result;
        for ([[maybe_unused]] auto i : range(num_samples))
        {
            auto sampled = select_molecule(rng_);
            EXPECT_LT(sampled.molecule, MucfMuonicMolecule::size_);
            result[sampled.molecule]++;
        }
        return result;
    }

  protected:
    HostCRef<DTMixMucfData> data_;
    Engine rng_;
    CycleRatesArray cycle_rates_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(MuonicMoleculeSelectorTest, deuterium_spin_selection)
{
    size_type const num_samples{1};
    CycleRatesArray rates;

    //// DD molecule /////
    {
        // Make DT and TT rates zero
        rates[MMM::deuterium_tritium] = {0, 0};
        rates[MMM::tritium_tritium] = {0, 0};

        // Test DD, F = 1/2
        rates[MMM::deuterium_deuterium] = {1e6, 0};
        auto const dd_1_result = MuonicMoleculeSelector(
            MMA::deuterium, HalfSpinInt{1}, rates)(rng_);

        EXPECT_EQ(dd_1_result.molecule, MMM::deuterium_deuterium);
        EXPECT_SOFT_EQ(dd_1_result.cycle_time, 1.9989533630516e-06);

        // Test DD, F = 3/2
        rates[MMM::deuterium_deuterium] = {0, 1e6};
        auto const dd_3_result = MuonicMoleculeSelector(
            MMA::deuterium, HalfSpinInt{3}, rates)(rng_);

        EXPECT_EQ(dd_3_result.molecule, MMM::deuterium_deuterium);
        EXPECT_SOFT_EQ(dd_3_result.cycle_time, 1.8031326676554e-07);
    }

    //// DT molecule /////
    {
        // Make DD and TT rates zero
        rates[MMM::deuterium_deuterium] = {0, 0};
        rates[MMM::tritium_tritium] = {0, 0};

        // Test DT, F = 0
        rates[MMM::deuterium_tritium] = {1e6, 0};
        auto const dt_0_result = MuonicMoleculeSelector(
            MMA::deuterium, HalfSpinInt{0}, rates)(rng_);

        EXPECT_EQ(dt_0_result.molecule, MMM::deuterium_tritium);
        EXPECT_SOFT_EQ(dt_0_result.cycle_time, 1.9989533630516e-06);

        // Test DT, F = 1
        rates[MMM::deuterium_tritium] = {0, 1e6};
        auto const dt_1_result = MuonicMoleculeSelector(
            MMA::deuterium, HalfSpinInt{1}, rates)(rng_);

        EXPECT_EQ(dt_1_result.molecule, MMM::deuterium_tritium);
        EXPECT_SOFT_EQ(dt_1_result.cycle_time, 1.8031326676554e-07);
    }
}

//---------------------------------------------------------------------------//
TEST_F(MuonicMoleculeSelectorTest, tritium_spin_selection) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
