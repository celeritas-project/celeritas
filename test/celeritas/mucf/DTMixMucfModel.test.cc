//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/DTMixMucfModel.test.cc
//---------------------------------------------------------------------------//

#include "celeritas/mucf/model/DTMixMucfModel.hh"

#include "corecel/io/Logger.hh"

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
    void SetUp() override
    {
        particles_ = this->particle_params();
        this->set_material("hdt_fuel");
        materials_ = this->material_params();
    }

  protected:
    std::shared_ptr<ParticleParams const> particles_;
    std::shared_ptr<MaterialParams const> materials_;
};

//---------------------------------------------------------------------------//
TEST_F(DTMixMucfModelTest, data)
{
    using Molecule = MucfMuonicMolecule;

    auto model = DTMixMucfModel(ActionId{0}, *particles_, *materials_);
    auto const& data = model.host_ref();
    auto const& pids = data.particle_ids;
    auto const& masses = data.particle_masses;

#define EXPECT_PDG_EQ(MEMBER) \
    EXPECT_EQ(pdg::MEMBER().get(), particles_->id_to_pdg(pids.MEMBER).get())

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
    EXPECT_EQ(masses.MEMBER, particles_->get(pids.MEMBER).mass())

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

    auto const& cycles = data.cycle_times;
    // DD cycle times
    EXPECT_SOFT_EQ(0, cycles[MuCfMatId{0}][Molecule::deuterium_deuterium][0]);
    EXPECT_SOFT_EQ(0, cycles[MuCfMatId{0}][Molecule::deuterium_deuterium][1]);
    // DT cycle times
    EXPECT_SOFT_EQ(1, cycles[MuCfMatId{0}][Molecule::deuterium_tritium][0]);
    EXPECT_SOFT_EQ(2, cycles[MuCfMatId{0}][Molecule::deuterium_tritium][1]);
    // TT cycle times
    EXPECT_SOFT_EQ(0, cycles[MuCfMatId{0}][Molecule::tritium_tritium][0]);
    EXPECT_SOFT_EQ(0, cycles[MuCfMatId{0}][Molecule::tritium_tritium][1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
