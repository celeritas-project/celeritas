//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/CutoffParams.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/CutoffParams.hh"

#include "corecel/cont/Range.hh"
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
        using namespace units;

        // Set up MaterialParams
        MaterialParams::Input m_input;
        m_input.elements = {
            {AtomicNumber{1}, AmuMass{1.008}, "H"},
            {AtomicNumber{13}, AmuMass{26.9815385}, "Al"},
            {AtomicNumber{11}, AmuMass{22.98976928}, "Na"},
            {AtomicNumber{53}, AmuMass{126.90447}, "I"},
        };
        m_input.materials = {
            // Sodium iodide
            {2.948915064677e+22,
             293.0,
             MatterState::solid,
             {{ElementId{2}, 0.5}, {ElementId{3}, 0.5}},
             "NaI"},
            // Void
            {0, 0, MatterState::unspecified, {}, "hard vacuum"},
            // Diatomic hydrogen
            {1.0739484359044669e+20,
             100.0,
             MatterState::gas,
             {{ElementId{0}, 1.0}},
             "H2"},
        };
        materials = std::make_shared<MaterialParams>(std::move(m_input));

        // Set up ParticleParams
        ParticleParams::Input p_input;
        constexpr auto zero = zero_quantity();
        constexpr auto stable = ParticleRecord::stable_decay_constant();

        p_input.push_back({"electron",
                           pdg::electron(),
                           MevMass{0.5109989461},
                           ElementaryCharge{-1},
                           stable});
        p_input.push_back({"gamma", pdg::gamma(), zero, zero, stable});
        p_input.push_back({"positron",
                           pdg::positron(),
                           MevMass{0.5109989461},
                           ElementaryCharge{1},
                           stable});
        p_input.push_back({"proton",
                           pdg::proton(),
                           MevMass{938.27208816},
                           ElementaryCharge{1},
                           stable});
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

    std::vector<double> energies, ranges;
    for (auto const pid : range(ParticleId{particles->size()}))
    {
        for (auto const mid : range(MaterialId{materials->size()}))
        {
            CutoffView cutoffs(cutoff.host_ref(), mid);
            if (pid == particles->find(pdg::proton()))
            {
                // Protons aren't currently used
                EXPECT_THROW(cutoffs.energy(pid), DebugError);
                EXPECT_THROW(cutoffs.range(pid), DebugError);
            }
            else
            {
                energies.push_back(cutoffs.energy(pid).value());
                ranges.push_back(cutoffs.range(pid));
            }
        }
    }

    double const expected_energies[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double const expected_ranges[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
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

    std::vector<double> energies, ranges;
    for (auto const pid : range(ParticleId{particles->size()}))
    {
        for (auto const mid : range(MaterialId{materials->size()}))
        {
            CutoffView cutoffs(cutoff.host_ref(), mid);
            if (pid == particles->find(pdg::proton()))
            {
                // Protons aren't currently used
                EXPECT_THROW(cutoffs.energy(pid), DebugError);
                EXPECT_THROW(cutoffs.range(pid), DebugError);
            }
            else
            {
                energies.push_back(cutoffs.energy(pid).value());
                ranges.push_back(cutoffs.range(pid));
            }
        }
    }

    double const expected_energies[] = {0.2, 0, 0.4, 0, 0, 0, 0, 0, 0};
    double const expected_ranges[] = {0.1, 0, 0.3, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}

TEST_F(CutoffParamsTest, apply_cuts)
{
    CutoffParams::Input input;
    input.materials = materials;
    input.particles = particles;
    input.cutoffs.insert({pdg::electron(), {{Energy{6}, 0.6}, {}, {}}});
    input.cutoffs.insert({pdg::gamma(), {{Energy{4}, 0.4}, {}, {}}});
    input.cutoffs.insert({pdg::positron(), {{Energy{2}, 0.2}, {}, {}}});
    input.apply_cuts = true;
    CutoffParams cutoff(input);

    CutoffView cutoffs(cutoff.host_ref(), MaterialId{0});
    EXPECT_TRUE(cutoffs.apply_cuts());

    Secondary secondary;
    secondary.energy = Energy{7};
    secondary.particle_id = particles->find(pdg::electron());
    EXPECT_FALSE(cutoffs.apply_cut(secondary));
    secondary.energy = Energy{5};
    EXPECT_TRUE(cutoffs.apply_cut(secondary));

    secondary.particle_id = particles->find(pdg::gamma());
    EXPECT_FALSE(cutoffs.apply_cut(secondary));
    secondary.energy = Energy{3};
    EXPECT_TRUE(cutoffs.apply_cut(secondary));

    secondary.particle_id = particles->find(pdg::positron());
    EXPECT_FALSE(cutoffs.apply_cut(secondary));
    secondary.energy = Energy{1};
    EXPECT_TRUE(cutoffs.apply_cut(secondary));
}

//---------------------------------------------------------------------------//

class CutoffParamsImportTest : public RootTestBase
{
  protected:
    char const* geometry_basename() const final { return "four-steel-slabs"; }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
};

TEST_F(CutoffParamsImportTest, import_cutoffs)
{
    std::vector<double> energies, ranges;
    for (auto const pid : {this->particle()->find(pdg::electron()),
                           this->particle()->find(pdg::gamma()),
                           this->particle()->find(pdg::positron())})
    {
        for (auto const mid : range(MaterialId{this->material()->size()}))
        {
            CutoffView cutoffs(this->cutoff()->host_ref(), mid);
            energies.push_back(cutoffs.energy(pid).value());
            ranges.push_back(cutoffs.range(pid));
            EXPECT_FALSE(cutoffs.apply_cuts());
        }
    }

    static double const expected_energies[] = {0.00099,
                                               1.3082781553076,
                                               0.00099,
                                               0.020822442086622,
                                               0.00099,
                                               1.2358930791935};
    static double const expected_ranges[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
