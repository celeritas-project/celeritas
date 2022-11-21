//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/RootImporter.hh"

#include <algorithm>
#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//
/*!
 * The \e four-steel-slabs.root is created by the \e app/celer-export-geant
 * using the \e four-steel-slabs.gdml example file available in \e app/data .
 *
 * In order to keep the \e four-steel-slabs.root file small, the stored file in
 * test/celeritas/data is exported using
 * GeantImporter::DataSelection::reader_data = false
 * in \e app/celer-export-geant .
 *
 * This test only checks if the loaded ROOT file is minimally correct. Detailed
 * verification of the imported data is done by \c GeantImporter.test .
 */
class RootImporterTest : public Test
{
  protected:
    void SetUp() override
    {
        root_filename_
            = this->test_data_path("celeritas", "four-steel-slabs.root");

        RootImporter import_from_root(root_filename_.c_str());
        data_ = import_from_root();
    }

    std::string root_filename_;
    ImportData  data_;

    ScopedRootErrorHandler scoped_root_error_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, particles)
{
    const auto& particles = data_.particles;
    EXPECT_EQ(3, particles.size());

    // Check all names/PDG codes
    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (const auto& particle : particles)
    {
        loaded_names.push_back(particle.name);
        loaded_pdgs.push_back(particle.pdg);
    }

    // Particle ordering is the same as in the ROOT file
    // clang-format off
    static const char* expected_loaded_names[] = {"e+", "e-", "gamma"};

    static const int expected_loaded_pdgs[] = {-11, 11, 22};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, elements)
{
    const auto& elements = data_.elements;
    EXPECT_EQ(4, elements.size());

    std::vector<std::string> names;
    for (const auto& element : elements)
    {
        names.push_back(element.name);
    }

    static const char* expected_names[] = {"Fe", "Cr", "Ni", "H"};
    EXPECT_VEC_EQ(expected_names, names);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, materials)
{
    const auto& materials = data_.materials;
    EXPECT_EQ(2, materials.size());

    std::vector<std::string> names;
    for (const auto& material : materials)
    {
        names.push_back(material.name);
    }

    static const char* expected_names[] = {"G4_Galactic", "G4_STAINLESS-STEEL"};
    EXPECT_VEC_EQ(expected_names, names);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, processes)
{
    const auto& processes = data_.processes;
    EXPECT_EQ(10, processes.size());

    auto find_process = [&processes](PDGNumber pdg, ImportProcessClass ipc) {
        return std::find_if(processes.begin(),
                            processes.end(),
                            [&pdg, &ipc](const ImportProcess& proc) {
                                return PDGNumber{proc.particle_pdg} == pdg
                                       && proc.process_class == ipc;
                            });
    };

    auto ioni = find_process(pdg::electron(), ImportProcessClass::e_ioni);
    ASSERT_NE(processes.end(), ioni);

    EXPECT_EQ(ImportProcessType::electromagnetic, ioni->process_type);
    ASSERT_EQ(1, ioni->models.size());
    EXPECT_EQ(ImportModelClass::moller_bhabha, ioni->models.front());
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, volumes)
{
    const auto& volumes = data_.volumes;
    EXPECT_EQ(5, volumes.size());

    std::vector<unsigned int> material_ids;
    std::vector<std::string>  names, solids;

    for (const auto& volume : volumes)
    {
        material_ids.push_back(volume.material_id);
        names.push_back(volume.name);
        solids.push_back(volume.solid_name);
    }

    const unsigned int expected_material_ids[] = {1, 1, 1, 1, 0};

    static const char* expected_names[] = {"box0x125555be0",
                                           "box0x125556d20",
                                           "box0x125557160",
                                           "box0x1255575a0",
                                           "World0x125555f10"};

    static const char* expected_solids[] = {"box0x125555b70",
                                            "box0x125556c70",
                                            "box0x1255570a0",
                                            "box0x125557500",
                                            "World0x125555ea0"};

    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_solids, solids);
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
