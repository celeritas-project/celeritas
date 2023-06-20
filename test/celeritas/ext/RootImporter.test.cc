//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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
    std::string_view geometry_basename() const { return "four-steel-slabs"sv; }

    ImportData const& imported_data() const;
};

//---------------------------------------------------------------------------//
auto RootImporterTest::imported_data() const -> ImportData const&
{
    static struct
    {
        std::string geometry_basename;
        ImportData imported;
    } i;
    auto geo_basename = this->geometry_basename();
    if (i.geometry_basename != geo_basename)
    {
        ScopedRootErrorHandler scoped_root_error;

        i.geometry_basename = geo_basename;
        std::string root_inp = this->test_data_path(
            "celeritas", (i.geometry_basename + ".root").c_str());

        RootImporter import(root_inp.c_str());
        i.imported = import();
    }
    CELER_ENSURE(!i.imported.particles.empty());
    return i.imported;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, particles)
{
    auto const& particles = this->imported_data().particles;
    EXPECT_EQ(3, particles.size());

    // Check all names/PDG codes
    std::vector<std::string> loaded_names;
    std::vector<int> loaded_pdgs;
    for (auto const& particle : particles)
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
    auto const& elements = this->imported_data().elements;
    EXPECT_EQ(4, elements.size());

    std::vector<std::string> names;
    for (auto const& element : elements)
    {
        names.push_back(element.name);
    }

    static char const* expected_names[] = {"Fe", "Cr", "Ni", "H"};
    EXPECT_VEC_EQ(expected_names, names);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, materials)
{
    auto const& materials = this->imported_data().materials;
    EXPECT_EQ(2, materials.size());

    std::vector<std::string> names;
    for (auto const& material : materials)
    {
        names.push_back(material.name);
    }

    static char const* expected_names[] = {"G4_Galactic", "G4_STAINLESS-STEEL"};
    EXPECT_VEC_EQ(expected_names, names);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, processes)
{
    auto const& processes = this->imported_data().processes;
    EXPECT_EQ(9, processes.size());

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
    EXPECT_EQ(ImportModelClass::moller_bhabha,
              ioni->models.front().model_class);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, volumes)
{
    auto const& volumes = this->imported_data().volumes;
    EXPECT_EQ(5, volumes.size());

    std::vector<unsigned int> material_ids;
    std::vector<std::string> names, solids;

    for (auto const& volume : volumes)
    {
        material_ids.push_back(volume.material_id);
        names.push_back(volume.name);
        solids.push_back(volume.solid_name);
    }

    unsigned int const expected_material_ids[] = {1, 1, 1, 1, 0};

    static char const* expected_names[] = {"box0x125555be0",
                                           "box0x125556d20",
                                           "box0x125557160",
                                           "box0x1255575a0",
                                           "World0x125555f10"};

    static char const* expected_solids[] = {"box0x125555b70",
                                            "box0x125556c70",
                                            "box0x1255570a0",
                                            "box0x125557500",
                                            "World0x125555ea0"};

    EXPECT_VEC_EQ(expected_material_ids, material_ids);
    EXPECT_VEC_EQ(expected_names, names);
    EXPECT_VEC_EQ(expected_solids, solids);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
