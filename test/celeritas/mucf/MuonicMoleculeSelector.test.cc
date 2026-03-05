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
#include "celeritas/mucf/Types.hh"
#include "celeritas/mucf/executor/detail/MuonicAtomSelector.hh"
#include "celeritas/mucf/executor/detail/MuonicAtomSpinSelector.hh"

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

    void SetUp() override
    {
        data_ = this->host_data();
        rng_.reset_count();

        mucfmatid_ = MucfMatId{0};
        cycle_rates_ = data_.cycle_rates[mucfmatid_];
    }

  protected:
    HostCRef<DTMixMucfData> data_;
    Engine rng_;
    MucfMatId mucfmatid_;
    CycleRatesArray cycle_rates_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(MuonicMoleculeSelectorTest, TEST_IF_CELERITAS_DOUBLE(dd_spin_selection))
{
    CycleRatesArray rates;

    // Make DT and TT rates zero
    rates[MMM::deuterium_tritium] = {0, 0};
    rates[MMM::tritium_tritium] = {0, 0};

    // DD, F = 1/2
    rates[MMM::deuterium_deuterium] = {1e6, 0};
    auto const dd_1_result
        = MuonicMoleculeSelector(MMA::deuterium, HalfSpinInt{1}, rates)(rng_);

    EXPECT_EQ(dd_1_result.molecule, MMM::deuterium_deuterium);
    EXPECT_SOFT_EQ(dd_1_result.cycle_time, 1.9989533630516e-06);

    // DD, F = 3/2
    rates[MMM::deuterium_deuterium] = {0, 1e6};
    auto const dd_3_result
        = MuonicMoleculeSelector(MMA::deuterium, HalfSpinInt{3}, rates)(rng_);

    EXPECT_EQ(dd_3_result.molecule, MMM::deuterium_deuterium);
    EXPECT_SOFT_EQ(dd_3_result.cycle_time, 1.8031326676554e-07);
}

//---------------------------------------------------------------------------//
TEST_F(MuonicMoleculeSelectorTest, TEST_IF_CELERITAS_DOUBLE(dt_spin_selection))
{
    CycleRatesArray rates;

    // Make DD and TT rates zero
    rates[MMM::deuterium_deuterium] = {0, 0};
    rates[MMM::tritium_tritium] = {0, 0};

    // DT, F = 0
    rates[MMM::deuterium_tritium] = {1e6, 0};
    auto const dt_0_result
        = MuonicMoleculeSelector(MMA::tritium, HalfSpinInt{0}, rates)(rng_);

    EXPECT_EQ(dt_0_result.molecule, MMM::deuterium_tritium);
    EXPECT_SOFT_EQ(dt_0_result.cycle_time, 1.9989533630516e-06);

    // DT, F = 1
    rates[MMM::deuterium_tritium] = {0, 1e6};
    auto const dt_2_result
        = MuonicMoleculeSelector(MMA::tritium, HalfSpinInt{2}, rates)(rng_);

    EXPECT_EQ(dt_2_result.molecule, MMM::deuterium_tritium);
    EXPECT_SOFT_EQ(dt_2_result.cycle_time, 1.8031326676554e-07);
}

//---------------------------------------------------------------------------//
TEST_F(MuonicMoleculeSelectorTest, TEST_IF_CELERITAS_DOUBLE(tt_spin_selection))
{
    CycleRatesArray rates;

    // TT, F = 1/2
    rates[MMM::deuterium_deuterium] = {0, 0};
    rates[MMM::deuterium_tritium] = {0, 0};
    rates[MMM::tritium_tritium] = {1e6, 0};

    auto const tt_1_result
        = MuonicMoleculeSelector(MMA::tritium, HalfSpinInt{1}, rates)(rng_);
    EXPECT_EQ(tt_1_result.molecule, MMM::tritium_tritium);
    EXPECT_SOFT_EQ(tt_1_result.cycle_time, 1.9989533630516e-06);
}

//---------------------------------------------------------------------------//
TEST_F(MuonicMoleculeSelectorTest, model_data)
{
    using MoleculeCountArray = EnumArray<MucfMuonicMolecule, size_type>;

    // Mimic executor behavior: form atom, then form molecule
    size_type const num_samples = 10000;
    auto const& rates = data_.cycle_rates[mucfmatid_];

    MoleculeCountArray molecule_counts;
    for ([[maybe_unused]] auto i : range(num_samples))
    {
        // Form muonic atom
        auto muonic_atom = MuonicAtomSelector(
            data_.isotopic_fractions[mucfmatid_][MucfIsotope::deuterium])(rng_);
        auto atom_spin = MuonicAtomSpinSelector(muonic_atom)(rng_);

        // Form molecule that will call determine the interactor
        auto result
            = MuonicMoleculeSelector(muonic_atom, atom_spin, rates)(rng_);
        molecule_counts[result.molecule]++;
    }

    // From Acceleron's simulation with equivalent material: 50/50 d/t at 300K,
    // the fraction of each fusion call is: DD: 20.1%, DT: 78.9%, TT: 0.9%.
    // The counts on this test (20%, 79%, 0.2%) are close to Acceleron's.
    EXPECT_EQ(2044, molecule_counts[MMM::deuterium_deuterium]);
    EXPECT_EQ(7940, molecule_counts[MMM::deuterium_tritium]);
    EXPECT_EQ(16, molecule_counts[MMM::tritium_tritium]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
