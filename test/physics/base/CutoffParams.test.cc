//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParams.test.cc
//---------------------------------------------------------------------------//
#include "physics/base/CutoffParams.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/base/Units.hh"
#include "physics/material/MaterialParams.hh"
#include "physics/material/MaterialInterface.hh"
#include "physics/material/ElementView.hh"
#include "physics/material/Types.hh"
#include "physics/material/detail/Utils.hh"
#include "base/Range.hh"
#include "io/RootImporter.hh"
#include "io/ImportData.hh"

#include "celeritas_test.hh"

using celeritas::CutoffParams;
using celeritas::CutoffView;
using celeritas::ElementId;
using celeritas::ImportData;
using celeritas::MaterialId;
using celeritas::MaterialParams;
using celeritas::MatterState;
using celeritas::ParticleId;
using celeritas::ParticleParams;
using celeritas::range;
using celeritas::RootImporter;
using celeritas::units::AmuMass;
using celeritas::units::MevEnergy;

namespace pdg = celeritas::pdg;
using namespace celeritas::units;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CutoffParamsTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        // Set up MaterialParams
        MaterialParams::Input m_input;
        m_input.elements = {
            {1, AmuMass{1.008}, "H"},
            {13, AmuMass{26.9815385}, "Al"},
            {11, AmuMass{22.98976928}, "Na"},
            {53, AmuMass{126.90447}, "I"},
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
        material_params = std::make_shared<MaterialParams>(std::move(m_input));

        // Set up ParticleParams
        ParticleParams::Input p_input;
        constexpr auto        zero = celeritas::zero_quantity();
        constexpr auto stable = celeritas::ParticleDef::stable_decay_constant();

        p_input.push_back({"electron",
                           pdg::electron(),
                           MevMass{0.5109989461},
                           ElementaryCharge{-1},
                           stable});
        p_input.push_back({"gamma", pdg::gamma(), zero, zero, stable});
        particle_params = std::make_shared<ParticleParams>(std::move(p_input));
    }

    std::shared_ptr<MaterialParams> material_params;
    std::shared_ptr<ParticleParams> particle_params;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(CutoffParamsTest, empty_cutoffs)
{
    CutoffParams::Input input;
    input.materials = material_params;
    input.particles = particle_params;

    // input.cutoffs left empty
    CutoffParams cutoff_params(input);

    std::vector<double> energies, ranges;
    for (const auto pid : range(ParticleId{particle_params->size()}))
    {
        for (const auto matid : range(MaterialId{material_params->size()}))
        {
            CutoffView cutoff_view(cutoff_params.host_pointers(), pid, matid);
            energies.push_back(cutoff_view.energy().value());
            ranges.push_back(cutoff_view.range());
        }
    }

    const double expected_energies[] = {0, 0, 0, 0, 0, 0};
    const double expected_ranges[]   = {0, 0, 0, 0, 0, 0};

    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}

TEST_F(CutoffParamsTest, electron_cutoffs)
{
    CutoffParams::Input           input;
    CutoffParams::MaterialCutoffs mat_cutoffs;
    input.materials = material_params;
    input.particles = particle_params;
    mat_cutoffs.push_back({MevEnergy{0.2}, 0.1});
    mat_cutoffs.push_back({MevEnergy{0.0}, 0.0});
    mat_cutoffs.push_back({MevEnergy{0.4}, 0.3});
    input.cutoffs.insert({pdg::electron(), mat_cutoffs});

    CutoffParams cutoff_params(input);

    std::vector<double> energies, ranges;
    for (const auto pid : range(ParticleId{particle_params->size()}))
    {
        for (const auto matid : range(MaterialId{material_params->size()}))
        {
            CutoffView cutoff_view(cutoff_params.host_pointers(), pid, matid);

            energies.push_back(cutoff_view.energy().value());
            ranges.push_back(cutoff_view.range());
        }
    }

    const double expected_energies[] = {0.2, 0, 0.4, 0, 0, 0};
    const double expected_ranges[]   = {0.1, 0, 0.3, 0, 0, 0};

    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}

//---------------------------------------------------------------------------//
// IMPORT CUTOFF DATA TEST
//---------------------------------------------------------------------------//

class CutoffParamsImportTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        root_filename_ = this->test_data_path("io", "geant-exporter-data.root");
        RootImporter import_from_root(root_filename_.c_str());
        data_ = import_from_root("geant4_data", "ImportData");
        ;
    }
    std::string root_filename_;
    ImportData  data_;
};

TEST_F(CutoffParamsImportTest, TEST_IF_CELERITAS_USE_ROOT(import_cutoffs))
{
    const auto particles = ParticleParams::from_import(data_);
    const auto materials = MaterialParams::from_import(data_);
    const auto cutoffs   = CutoffParams::from_import(data_);

    std::vector<double> energies, ranges;

    for (const auto pid : range(ParticleId{particles->size()}))
    {
        for (const auto matid : range(MaterialId{materials->size()}))
        {
            CutoffView cutoff_view(cutoffs->host_pointers(), pid, matid);
            energies.push_back(cutoff_view.energy().value());
            ranges.push_back(cutoff_view.range());
        }
    }

    // clang-format off
    const double expected_energies[] = {0.00099, 0.0173344452484621, 0.00099,
        0.970694711604435, 0.00099, 0.926090152562135, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0.07, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0};
    const double expected_ranges[] = {0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_energies, energies);
    EXPECT_VEC_SOFT_EQ(expected_ranges, ranges);
}