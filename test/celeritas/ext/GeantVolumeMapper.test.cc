//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantVolumeMapper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantVolumeMapper.hh"

#include <regex>
#include <string>
#include <vector>

#include "corecel/Config.hh"

#include "geocel/GeantGeoUtils.hh"
#include "celeritas/geo/CoreGeoParams.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_GEANT4
#    include <G4LogicalVolume.hh>
#    include <G4Material.hh>
#    include <G4NistManager.hh>
#    include <G4Orb.hh>
#    include <G4PVPlacement.hh>
#    include <G4SubtractionSolid.hh>
#    include <G4ThreeVector.hh>
#    include <G4TransportationManager.hh>
#    include <G4VPhysicalVolume.hh>
#    include <G4VSolid.hh>

#endif
#if CELERITAS_USE_VECGEOM
#    include "geocel/vg/VecgeomParams.hh"
#endif

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"
#include "orange/OrangeInput.hh"
#include "orange/OrangeParams.hh"
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
    GeantVolumeMapperTestBase()
        : store_log_(&celeritas::world_logger(), LogLevel::warning)
    {
    }

    // Clean up geometry at destruction
    void TearDown() override
    {
        geant_geo_params_.reset();
        geo_params_.reset();
    }

    // Construct geometry
    void build()
    {
        if (CELERITAS_USE_GEANT4)
        {
            this->build_g4();
            CELER_ASSERT(geant_geo_params_);
            celeritas::geant_geo(geant_geo_params_);
        }
        CELER_ASSERT(!logical_.empty());

        this->build_vecgeom();
        this->build_orange();

        CELER_ENSURE(geo_params_);
    }

    G4VPhysicalVolume const& world() const
    {
        CELER_VALIDATE(!physical_.empty() && physical_.back(),
                       << "Geant4 world was not set up");
        return *physical_.front();
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
    std::shared_ptr<GeantGeoParams> geant_geo_params_;
    std::shared_ptr<CoreGeoParams> geo_params_;

    ScopedLogStorer store_log_;
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

    // Construct Geant4 geometry
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(
        physical_.front());
    geant_geo_params_ = GeantGeoParams::from_tracking_manager();
#    if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4
    geo_params_ = geant_geo_params_;
#    endif
#else
    CELER_NOT_CONFIGURED("Geant4");
#endif
}

void NestedTest::build_vecgeom()
{
    CELER_EXPECT(!physical_.empty());
#if CELERITAS_USE_VECGEOM
    CELER_ASSERT(geant_geo_params_);
    auto geo = VecgeomParams::from_geant(geant_geo_params_);
#else
    int geo;
    CELER_DISCARD(geo);
#endif
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM
    geo_params_ = std::move(geo);
#endif
}

void NestedTest::build_orange()
{
    // Create ORANGE input manually
    UnitInput ui;
    ui.label = "global";

    auto radius = static_cast<real_type>(names_.size() + 1);
    ui.bbox = {{-radius, -radius, -radius}, {radius, radius, radius}};

    LocalSurfaceId daughter;
    for (std::string const& name : names_)
    {
        // Insert surfaces
        LocalSurfaceId parent{static_cast<size_type>(ui.surfaces.size())};
        ui.surfaces.push_back(Sphere({0, 0, 0}, radius));
        radius -= 1.0;

        // Insert volume
        VolumeInput vi;
        vi.label = name;
        vi.zorder = ZOrder::media;
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
    input.universes.push_back(std::move(ui));
    input.tol = Tolerance<>::from_default();
    auto geo = std::make_shared<OrangeParams>(std::move(input));
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE
    geo_params_ = std::move(geo);
#else
    CELER_DISCARD(geo);
#endif
}

#if CELERITAS_CORE_GEO != CELERITAS_CORE_VECGEOM
#    define SKIP_UNLESS_VECGEOM(NAME) DISABLED_##NAME
#else
#    define SKIP_UNLESS_VECGEOM(NAME) NAME
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
        ASSERT_NE(VolumeId{}, vol_id)
            << "searching for " << PrintableLV{logical_[i]};
        EXPECT_EQ(names_[i], geo_params_->volumes().at(vol_id).name);
    }

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        static char const* const expected_log_messages[] = {
            R"(Failed to exactly match ORANGE volume from Geant4 volume 'world'; found 'world@global' by omitting the extension)",
            R"(Failed to exactly match ORANGE volume from Geant4 volume 'outer'; found 'outer@global' by omitting the extension)",
            R"(Failed to exactly match ORANGE volume from Geant4 volume 'middle'; found 'middle@global' by omitting the extension)",
            R"(Failed to exactly match ORANGE volume from Geant4 volume 'inner'; found 'inner@global' by omitting the extension)",
        };
        EXPECT_VEC_EQ(expected_log_messages, store_log_.messages());
        static char const* const expected_log_levels[]
            = {"warning", "warning", "warning", "warning"};
        EXPECT_VEC_EQ(expected_log_levels, store_log_.levels());
    }
    else
    {
        EXPECT_EQ(0, store_log_.messages().size()) << store_log_;
    }
}

// Geant4 constructed directly by user
TEST_F(NestedTest, SKIP_UNLESS_VECGEOM(duplicated))
{
    names_ = {"world", "dup", "dup", "bob"};
    this->build();
    CELER_ASSERT(logical_.size() == names_.size());

    GeantVolumeMapper find_vol{*geo_params_};
    for (auto i : range(names_.size()))
    {
        VolumeId vol_id = find_vol(*logical_[i]);
        ASSERT_NE(VolumeId{}, vol_id);
        EXPECT_EQ(names_[i], geo_params_->volumes().at(vol_id).name);
    }

    // IDs for the unique LVs should be different
    EXPECT_NE(find_vol(*logical_[1]), find_vol(*logical_[2]));

    if (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_GEANT4)
    {
        EXPECT_EQ(0, store_log_.messages().size());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
