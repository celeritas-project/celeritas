//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantScintillationLoader.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/GeantScintillationLoader.hh"

#include <string>
#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>
#include <G4SystemOfUnits.hh>

#include "celeritas/Units.hh"
#include "celeritas/ext/detail/GeantOpticalMatHelper.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

class GeantScintillationLoaderTest : public ::celeritas::test::Test
{
  protected:
    //! Create a material with a single-component scintillation spectrum
    G4Material* make_material(std::string const& name)
    {
        auto* mat = new G4Material(name, 18, 39.95 * g / mole, 1.4 * g / cm3);
        mpt_ = new G4MaterialPropertiesTable;
        mpt_->AddConstProperty("SCINTILLATIONYIELD", 100. / MeV);
        mpt_->AddConstProperty("SCINTILLATIONTIMECONSTANT1", 10 * ns);
        mpt_->AddConstProperty("SCINTILLATIONRISETIME1", 1 * ns);
        this->add_component("SCINTILLATIONCOMPONENT1");
        mat->SetMaterialPropertiesTable(mpt_);
        return mat;
    }

    void add_component(char const* name)
    {
        // Use the array overload, which is available for Geant4 10
        double energy[] = {2 * eV, 3 * eV};
        double value[] = {1, 1};
        mpt_->AddProperty(name, energy, value, 2);
    }

    inp::ScintillationMaterial load(G4Material const* mat)
    {
        inp::ScintillationProcess process;
        GeantScintillationLoader load_material{process};
        OptMatId const id{0};
        load_material(GeantOpticalMatHelper{id, mat});
        EXPECT_EQ(1, process.materials.size());
        return process.materials.at(id);
    }

    G4MaterialPropertiesTable* mpt_{nullptr};
};

//---------------------------------------------------------------------------//
// Geant4 uses a default yield of 1 for the first component
TEST_F(GeantScintillationLoaderTest, default_component_yield)
{
    auto* mat = this->make_material("one_component");

    auto scint = this->load(mat);
    ASSERT_EQ(1, scint.components.size());
    EXPECT_SOFT_EQ(100, scint.components[0].yield);
}

//---------------------------------------------------------------------------//
// Geant4 uses a default yield of 0 for the second and third components
TEST_F(GeantScintillationLoaderTest, default_zero_yield)
{
    auto* mat = this->make_material("two_components");
    mpt_->AddConstProperty("SCINTILLATIONTIMECONSTANT2", 100 * ns);
    this->add_component("SCINTILLATIONCOMPONENT2");

    auto scint = this->load(mat);
    ASSERT_EQ(1, scint.components.size());
    EXPECT_SOFT_EQ(100, scint.components[0].yield);
}

//---------------------------------------------------------------------------//
// The loader always reads the rise time from the material properties: the
// Geant4 finite rise time setting is applied by GeantPhysicsLoader
TEST_F(GeantScintillationLoaderTest, rise_time)
{
    auto* mat = this->make_material("rise_time");

    auto scint = this->load(mat);
    ASSERT_EQ(1, scint.components.size());
    EXPECT_SOFT_EQ(1 * units::nanosecond, scint.components[0].rise_time);
    EXPECT_SOFT_EQ(10 * units::nanosecond, scint.components[0].fall_time);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
