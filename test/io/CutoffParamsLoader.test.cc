//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParamsLoader.test.cc
//---------------------------------------------------------------------------//
#include "io/CutoffParamsLoader.hh"

#include "io/MaterialParamsLoader.hh"
#include "io/ParticleParamsLoader.hh"
#include "physics/base/CutoffView.hh"

#include "celeritas_test.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CutoffParamsLoaderTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        root_filename_ = this->test_data_path("io", "geant-exporter-data.root");
    }
    std::string root_filename_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(CutoffParamsLoaderTest, load_cutoff_params)
{
    RootLoader           root_loader(this->root_filename_.c_str());
    ParticleParamsLoader particle_loader(root_loader);
    MaterialParamsLoader material_loader(root_loader);
    CutoffParamsLoader   cutoff_loader(root_loader);

    const auto particles = particle_loader();
    const auto materials = material_loader();
    const auto cutoffs   = cutoff_loader();

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
