//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/DTMixMucfModel.test.cc
//---------------------------------------------------------------------------//

#include "celeritas/mucf/model/DTMixMucfModel.hh"

#include "MucfInteractorHostTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class DTMixMucfModelTest : public MucfInteractorHostTestBase
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
TEST_F(DTMixMucfModelTest, data)
{
    using Molecule = MucfMuonicMolecule;

    auto const& data = this->host_data();
    auto const& pids = data.particle_ids;
    auto const& masses = data.particle_masses;

#define EXPECT_PDG_EQ(MEMBER)      \
    EXPECT_EQ(pdg::MEMBER().get(), \
              this->particle_params()->id_to_pdg(pids.MEMBER).get())

    EXPECT_PDG_EQ(mu_minus);
    EXPECT_PDG_EQ(neutron);
    EXPECT_PDG_EQ(proton);
    EXPECT_PDG_EQ(alpha);
    EXPECT_PDG_EQ(he3);
    EXPECT_PDG_EQ(muonic_hydrogen);
    EXPECT_PDG_EQ(muonic_deuteron);
    EXPECT_PDG_EQ(muonic_triton);
    EXPECT_PDG_EQ(muonic_alpha);
    EXPECT_PDG_EQ(muonic_he3);

#undef EXPECT_PDG_EQ

#define EXPECT_MASS_EQ(MEMBER) \
    EXPECT_EQ(masses.MEMBER, this->particle_params()->get(pids.MEMBER).mass())

    EXPECT_MASS_EQ(mu_minus);
    EXPECT_MASS_EQ(neutron);
    EXPECT_MASS_EQ(proton);
    EXPECT_MASS_EQ(alpha);
    EXPECT_MASS_EQ(he3);
    EXPECT_MASS_EQ(muonic_hydrogen);
    EXPECT_MASS_EQ(muonic_deuteron);
    EXPECT_MASS_EQ(muonic_triton);
    EXPECT_MASS_EQ(muonic_alpha);
    EXPECT_MASS_EQ(muonic_he3);

#undef EXPECT_MASS_EQ

    EXPECT_EQ(21, data.muon_energy_cdf.grid.size());

    // Check sotopic fractions
    // Single material with 50/50 d and t fractions
    EXPECT_SOFT_EQ(
        0.5, data.isotopic_fractions[MuCfMatId{0}][MucfIsotope::deuterium]);
    EXPECT_SOFT_EQ(
        0.5, data.isotopic_fractions[MuCfMatId{0}][MucfIsotope::tritium]);

    // Cycle times are in seconds
    // DD (reactivity of F = 3/2 is almost negligible, with huge cycle times)
    static double const expected_dd_1_over_2_cycle_time{1.8312922823566493e-06};
    static double const expected_dd_3_over_2_cycle_time{1.1439517165483279};
    // DT
    static double const expected_dt_0_cycle_time{1.0182824459351898e-08};
    static double const expected_dt_1_cycle_time{5.098478246172425e-09};
    // TT
    static double const expected_tt_1_over_2_cycle_time{1.4056833511329384e-06};

    auto const& cycles = data.cycle_times;

    // DD cycle times
    EXPECT_SOFT_EQ(expected_dd_1_over_2_cycle_time,
                   cycles[MuCfMatId{0}][Molecule::deuterium_deuterium][0]);
    EXPECT_SOFT_EQ(expected_dd_3_over_2_cycle_time,
                   cycles[MuCfMatId{0}][Molecule::deuterium_deuterium][1]);
    // DT cycle times
    EXPECT_SOFT_EQ(expected_dt_0_cycle_time,
                   cycles[MuCfMatId{0}][Molecule::deuterium_tritium][0]);
    EXPECT_SOFT_EQ(expected_dt_1_cycle_time,
                   cycles[MuCfMatId{0}][Molecule::deuterium_tritium][1]);
    // TT cycle times
    EXPECT_SOFT_EQ(expected_tt_1_over_2_cycle_time,
                   cycles[MuCfMatId{0}][Molecule::tritium_tritium][0]);
    EXPECT_SOFT_EQ(0, cycles[MuCfMatId{0}][Molecule::tritium_tritium][1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
