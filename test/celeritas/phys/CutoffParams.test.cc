//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/CutoffParams.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/CutoffParams.hh"

#include "corecel/cont/Range.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/detail/Utils.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Secondary.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class CutoffParamsTest : public Test
{
  protected:
    using Energy = units::MevEnergy;

    void SetUp() override
    {
        using namespace constants;
        using namespace units;

        // Set up MaterialParams
        MaterialParams::Input m_input;
        m_input.elements = {
            {AtomicNumber{1}, AmuMass{1.008}, {}, "H"},
            {AtomicNumber{13}, AmuMass{26.9815385}, {}, "Al"},
            {AtomicNumber{11}, AmuMass{22.98976928}, {}, "Na"},
            {AtomicNumber{53}, AmuMass{126.90447}, {}, "I"},
        };
        m_input.materials = {
            // Sodium iodide
            {native_value_from(InvCcDensity{2.948915064677e+22}),
             293.0,
             MatterState::solid,
             {{ElementId{2}, 0.5}, {ElementId{3}, 0.5}},
             "NaI"},
            // Void
            {0, 0, MatterState::unspecified, {}, "hard vacuum"},
            // Diatomic hydrogen
            {native_value_from(InvCcDensity{1.0739484359044669e+20}),
             100.0,
             MatterState::gas,
             {{ElementId{0}, 1.0}},
             "H2"},
        };
        materials = std::make_shared<MaterialParams>(std::move(m_input));

        // Set up ParticleParams
        ParticleParams::Input p_input;
        constexpr auto zero = zero_quantity();

        p_input.push_back({"electron",
                           pdg::electron(),
                           MevMass{0.5109989461},
                           ElementaryCharge{-1},
                           stable_decay_constant});
        p_input.push_back(
            {"gamma", pdg::gamma(), zero, zero, stable_decay_constant});
        p_input.push_back({"positron",
                           pdg::positron(),
                           MevMass{0.5109989461},
                           ElementaryCharge{1},
                           stable_decay_constant});
        p_input.push_back({"proton",
                           pdg::proton(),
                           MevMass{938.27208816},
                           ElementaryCharge{1},
                           stable_decay_constant});
        particles = std::make_shared<ParticleParams>(std::move(p_input));
    }

    std::shared_ptr<MaterialParams> materials;
    std::shared_ptr<ParticleParams> particles;
};

TEST_F(CutoffParamsTest, empty_cutoffs)
{
    CutoffParams::Input input;
    input.materials = materials;
    input.particles = particles;

    // input.cutoffs left empty
    CutoffParams cutoff(input);

    std::vector<real_type> energies, ranges;
    for (auto const pid : range(ParticleId{particles->size()}))
    {
        for (auto const mid : range(PhysMatId{materials->size()}))
        {
            CutoffView cutoffs(cutoff.host_ref(), mid);
            if (pid != particles->find(pdg::proton()))
            {
                energies.push_back(cutoffs.energy(pid).value());
                ranges.push_back(cutoffs.range(pid));
            }
            else if (CELERITAS_DEBUG)
            {
                // Protons aren't currently used
                EXPECT_THROW(cutoffs.energy(pid), DebugError);
                EXPECT_THROW(cutoffs.range(pid), DebugError);
            }
        }
    }

    real_type const expected_energies[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    real_type const expected_ranges[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}

TEST_F(CutoffParamsTest, electron_cutoffs)
{
    CutoffParams::Input input;
    CutoffParams::MaterialCutoffs mat_cutoffs;
    input.materials = materials;
    input.particles = particles;
    mat_cutoffs.push_back({Energy{0.2}, 0.1});
    mat_cutoffs.push_back({Energy{0.0}, 0.0});
    mat_cutoffs.push_back({Energy{0.4}, 0.3});
    input.cutoffs.insert({pdg::electron(), mat_cutoffs});

    CutoffParams cutoff(input);

    std::vector<real_type> energies, ranges;
    for (auto const pid : range(ParticleId{particles->size()}))
    {
        for (auto const mid : range(PhysMatId{materials->size()}))
        {
            CutoffView cutoffs(cutoff.host_ref(), mid);
            if (pid != particles->find(pdg::proton()))
            {
                energies.push_back(cutoffs.energy(pid).value());
                ranges.push_back(cutoffs.range(pid));
            }
            else if (CELERITAS_DEBUG)
            {
                // Protons aren't currently used
                EXPECT_THROW(cutoffs.energy(pid), DebugError);
                EXPECT_THROW(cutoffs.range(pid), DebugError);
            }
        }
    }

    real_type const expected_energies[] = {0.2, 0, 0.4, 0, 0, 0, 0, 0, 0};
    real_type const expected_ranges[] = {0.1, 0, 0.3, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}

TEST_F(CutoffParamsTest, apply_post_interaction)
{
    CutoffParams::Input input;
    input.materials = materials;
    input.particles = particles;
    input.cutoffs.insert({pdg::electron(), {{Energy{6}, 0.6}, {}, {}}});
    input.cutoffs.insert({pdg::gamma(), {{Energy{4}, 0.4}, {}, {}}});
    input.cutoffs.insert({pdg::positron(), {{Energy{2}, 0.2}, {}, {}}});
    input.apply_post_interaction = true;
    CutoffParams cutoff(input);

    CutoffView cutoffs(cutoff.host_ref(), PhysMatId{0});
    EXPECT_TRUE(cutoffs.apply_post_interaction());

    Secondary secondary;
    secondary.energy = Energy{7};
    secondary.particle_id = particles->find(pdg::electron());
    EXPECT_FALSE(cutoffs.apply(secondary));
    secondary.energy = Energy{5};
    EXPECT_TRUE(cutoffs.apply(secondary));

    secondary.particle_id = particles->find(pdg::gamma());
    EXPECT_FALSE(cutoffs.apply(secondary));
    secondary.energy = Energy{3};
    EXPECT_TRUE(cutoffs.apply(secondary));

    secondary.particle_id = particles->find(pdg::positron());
    EXPECT_FALSE(cutoffs.apply(secondary));
    secondary.energy = Energy{1};
    EXPECT_TRUE(cutoffs.apply(secondary));
}

//---------------------------------------------------------------------------//

#define CutoffParamsImportTest \
    TEST_IF_CELERITAS_USE_ROOT(CutoffParamsImportTest)
class CutoffParamsImportTest : public RootTestBase
{
  protected:
    std::string_view gdml_basename() const override
    {
        return "four-steel-slabs"sv;
    }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
};

TEST_F(CutoffParamsImportTest, import_cutoffs)
{
    std::vector<real_type> energies, ranges;
    for (auto const pid : {this->particle()->find(pdg::electron()),
                           this->particle()->find(pdg::gamma()),
                           this->particle()->find(pdg::positron())})
    {
        for (auto const mid : range(PhysMatId{this->material()->size()}))
        {
            CutoffView cutoffs(this->cutoff()->host_ref(), mid);
            energies.push_back(cutoffs.energy(pid).value());
            ranges.push_back(to_cm(cutoffs.range(pid)));
            EXPECT_FALSE(cutoffs.apply_post_interaction());
        }
    }

    static real_type const expected_energies[] = {0.00099,
                                                  1.3082781553076,
                                                  0.00099,
                                                  0.020822442086622,
                                                  0.00099,
                                                  1.2358930791935};
    static real_type const expected_ranges[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
