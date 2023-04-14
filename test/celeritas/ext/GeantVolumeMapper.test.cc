//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantVolumeMapper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantVolumeMapper.hh"

#include <regex>
#include <string>
#include <vector>

#include "celeritas_config.h"
#include "celeritas/geo/GeoParams.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_GEANT4
#    include <G4LogicalVolume.hh>
#    include <G4LogicalVolumeStore.hh>
#    include <G4Material.hh>
#    include <G4NistManager.hh>
#    include <G4Orb.hh>
#    include <G4PVPlacement.hh>
#    include <G4PhysicalVolumeStore.hh>
#    include <G4SolidStore.hh>
#    include <G4SubtractionSolid.hh>
#    include <G4ThreeVector.hh>
#    include <G4VPhysicalVolume.hh>
#    include <G4VSolid.hh>
#endif

#include "corecel/io/Logger.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "orange/OrangeParams.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/construct/SurfaceInputBuilder.hh"
#include "orange/surf/Sphere.hh"

class G4VSolid;
class G4LogicalVolume;
class G4VPhysicalVolume;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class GeantVolumeMapperTestBase : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        using namespace std::placeholders;
        celeritas::world_logger() = Logger(
            MpiCommunicator{},
            std::bind(
                &GeantVolumeMapperTestBase::log_message, this, _1, _2, _3));
    }

    void log_message(Provenance, LogLevel lev, std::string msg)
    {
        if (lev > LogLevel::info)
        {
            static const std::regex delete_ansi("\033\\[[0-9;]*m");
            messages_.push_back(std::regex_replace(msg, delete_ansi, ""));
        }
    }

    static void TearDownTestCase()
    {
        // Restore logger
        celeritas::world_logger() = celeritas::make_default_world_logger();
    }

    // Clean up geometry at destruction
    void TearDown() override
    {
#if CELERITAS_USE_GEANT4
        G4PhysicalVolumeStore::Clean();
        G4LogicalVolumeStore::Clean();
        G4SolidStore::Clean();
#endif
        geo_params_.reset();
    }

    // Construct geometry
    void build()
    {
        if (CELERITAS_USE_GEANT4)
        {
            this->build_g4();
        }
        CELER_ASSERT(!logical_.empty());

        if (CELERITAS_USE_VECGEOM)
        {
            this->build_vecgeom();
        }
        else
        {
            this->build_orange();
        }
        CELER_ENSURE(geo_params_);
    }

  private:
    virtual void build_g4() = 0;
    virtual void build_vecgeom() = 0;
    virtual void build_orange() = 0;

  protected:
    // Non-owning pointers
    std::vector<G4VSolid*> solids_;
    std::vector<G4LogicalVolume*> logical_;
    std::vector<G4VPhysicalVolume*> physical_;

    // Celeritas data
    std::shared_ptr<GeoParams> geo_params_;
    std::vector<std::string> messages_;
};

//---------------------------------------------------------------------------//
// NESTED TEST
//---------------------------------------------------------------------------//
class NestedTest : public GeantVolumeMapperTestBase
{
  private:
    void build_g4() final;
    void build_vecgeom() final;
    void build_orange() final;

  protected:
    std::vector<std::string> names_;
};

void NestedTest::build_g4()
{
    CELER_EXPECT(!names_.empty());
#if CELERITAS_USE_GEANT4
    G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");

    G4LogicalVolume* parent_lv = nullptr;
    double radius{static_cast<double>(names_.size()) + 1};
    for (std::string const& name : names_)
    {
        // Create solid shape
        solids_.push_back(new G4Orb(name + "_solid", radius));

        // Create logical volume
        logical_.push_back(new G4LogicalVolume(solids_.back(), mat, name));

        // Create physical volume
        physical_.push_back(new G4PVPlacement(G4Transform3D{},
                                              logical_.back(),
                                              name + "_pv",
                                              parent_lv,
                                              /* pMany = */ false,
                                              /* pCopyNo = */ 0));
        radius -= 1.0;
        parent_lv = logical_.back();
    }

    CELER_ASSERT(mat);
#else
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

void NestedTest::build_vecgeom()
{
    CELER_EXPECT(!physical_.empty());
    geo_params_ = std::make_shared<GeoParams>(physical_.front());
}

void NestedTest::build_orange()
{
    // Create ORANGE input manually
    UnitInput ui;
    ui.label = "global";

    double radius{static_cast<double>(names_.size()) + 1};
    ui.bbox = {{-radius, -radius, -radius}, {radius, radius, radius}};

    SurfaceInputBuilder insert_surface(&ui.surfaces);
    LocalSurfaceId daughter;
    for (std::string const& name : names_)
    {
        // Insert surfaces
        auto parent = insert_surface(Sphere({0, 0, 0}, radius), Label("name"));
        radius -= 1.0;

        // Insert volume
        VolumeInput vi;
        vi.label = name;
        if (daughter)
        {
            vi.logic = {1, logic::lnot, 0, logic::land};
            vi.faces = {daughter, parent};
        }
        else
        {
            vi.logic = {0, logic::lnot};
            vi.faces = {parent};
        }
        ui.volumes.push_back(std::move(vi));
        daughter = parent;
    }

    OrangeInput input;
    input.max_level = 1;
    input.universes.push_back(std::move(ui));
    auto geo = std::make_shared<OrangeParams>(std::move(input));
#if !CELERITAS_USE_VECGEOM
    geo_params_ = std::move(geo);
#else
    (void)sizeof(geo);
#endif
    CELER_ENSURE(geo_params_);
}

//---------------------------------------------------------------------------//
// NESTED TEST
//---------------------------------------------------------------------------//
class IntersectionTest : public GeantVolumeMapperTestBase
{
  private:
    void build_g4() final;
    void build_vecgeom() final;
    void build_orange() final;

  protected:
    bool suffix_{false};
};

#if !CELERITAS_USE_VECGEOM
#    define SKIP_IF_ORANGE(NAME) DISABLED_##NAME
#else
#    define SKIP_IF_ORANGE(NAME) NAME
#endif

//---------------------------------------------------------------------------//
// Geant4 constructed directly by user
TEST_F(NestedTest, unique)
{
    names_ = {"world", "outer", "middle", "inner"};
    this->build();
    CELER_ASSERT(logical_.size() == names_.size());

    GeantVolumeMapper find_vol{*geo_params_};
    for (auto i : range(names_.size()))
    {
        VolumeId vol_id = find_vol(*logical_[i]);
        ASSERT_NE(VolumeId{}, vol_id);
        EXPECT_EQ(names_[i], geo_params_->id_to_label(vol_id).name);
    }

    if (CELERITAS_USE_VECGEOM)
    {
        EXPECT_EQ(0, messages_.size());
    }
    else
    {
        static std::string const expected_messages[]
            = {"Failed to exactly match ORANGE volume from Geant4 volume "
               "'world'; found 'world@global' by omitting the extension",
               "Failed to exactly match ORANGE volume from Geant4 volume "
               "'outer'; found 'outer@global' by omitting the extension",
               "Failed to exactly match ORANGE volume from Geant4 volume "
               "'middle'; found 'middle@global' by omitting the extension",
               "Failed to exactly match ORANGE volume from Geant4 volume "
               "'inner'; found 'inner@global' by omitting the extension"};
        EXPECT_VEC_EQ(expected_messages, messages_);
    }
}

// Geant4 constructed directly by user
TEST_F(NestedTest, SKIP_IF_ORANGE(duplicated))
{
    names_ = {"world", "dup", "dup", "bob"};
    this->build();
    CELER_ASSERT(logical_.size() == names_.size());

    GeantVolumeMapper find_vol{*geo_params_};
    for (auto i : range(names_.size()))
    {
        VolumeId vol_id = find_vol(*logical_[i]);
        ASSERT_NE(VolumeId{}, vol_id);
        EXPECT_EQ(names_[i], geo_params_->id_to_label(vol_id).name);
    }

    // IDs for the unique LVs should be different
    EXPECT_NE(find_vol(*logical_[1]), find_vol(*logical_[2]));

    EXPECT_EQ(0, messages_.size());
}

// Geant4 constructed from celeritas::LoadGdml (no stripping)
TEST_F(NestedTest, SKIP_IF_ORANGE(suffixed))
{
    names_ = {"world0xabc123", "outer0x123", "inner0xabc"};
    this->build();
    CELER_ASSERT(logical_.size() == names_.size());

    GeantVolumeMapper find_vol{*geo_params_};
    for (auto i : range(names_.size()))
    {
        VolumeId vol_id = find_vol(*logical_[i]);
        ASSERT_NE(VolumeId{}, vol_id);
        EXPECT_EQ(Label::from_geant(names_[i]),
                  geo_params_->id_to_label(vol_id));
    }
}

// Loaded GDML through demo app without stripping, then not stripped again
TEST_F(NestedTest, SKIP_IF_ORANGE(duplicated_suffixed))
{
    names_ = {"world0x1", "dup0x2", "dup0x3", "bob0x4"};
    this->build();
    CELER_ASSERT(logical_.size() == names_.size());

    GeantVolumeMapper find_vol{*geo_params_};
    for (auto i : range(names_.size()))
    {
        VolumeId vol_id = find_vol(*logical_[i]);
        ASSERT_NE(VolumeId{}, vol_id);
        EXPECT_EQ(Label::from_geant(names_[i]),
                  geo_params_->id_to_label(vol_id));
    }
}

TEST_F(NestedTest, SKIP_IF_ORANGE(double_prefixed))
{
    names_ = {"world0x10xa", "outer0x20xb", "inner0x30xc"};
    this->build();
    CELER_ASSERT(logical_.size() == names_.size());

    GeantVolumeMapper find_vol{*geo_params_};
    for (auto i : range(names_.size()))
    {
        VolumeId vol_id = find_vol(*logical_[i]);
        ASSERT_NE(VolumeId{}, vol_id);
        EXPECT_EQ(Label::from_geant(names_[i]),
                  geo_params_->id_to_label(vol_id));
    }
}

TEST_F(NestedTest, SKIP_IF_ORANGE(duplicated_double_prefixed))
{
    names_ = {"world0x10xa", "dup0x20xb", "dup0x30xc", "bob0x40xd"};
    this->build();
    CELER_ASSERT(logical_.size() == names_.size());

    GeantVolumeMapper find_vol{*geo_params_};
    for (auto i : range(names_.size()))
    {
        VolumeId vol_id = find_vol(*logical_[i]);
        ASSERT_NE(VolumeId{}, vol_id);
        EXPECT_EQ(Label::from_geant(names_[i]),
                  geo_params_->id_to_label(vol_id));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
