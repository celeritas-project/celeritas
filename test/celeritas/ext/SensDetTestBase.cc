//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/SensDetTestBase.cc
//---------------------------------------------------------------------------//
#include "SensDetTestBase.hh"

#include <G4LogicalVolumeStore.hh>
#include <G4SDManager.hh>

#include "corecel/Config.hh"

#include "corecel/io/Join.hh"

#include "SimpleSensitiveDetector.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Attach SDs when building geometry
auto SensDetTestBase::build_geant_geo(std::string const& filename) const
    -> SPConstGeantGeo
{
    CELER_EXPECT(detectors_.empty());

    // Construct geo
    auto result = Base::build_geant_geo(filename);

    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    CELER_ASSERT(lv_store && !lv_store->empty());

    // Attach SDs
    auto sd_vol_names = this->detector_volumes();

    // Find and set up sensitive detectors
    G4SDManager* sd_manager = G4SDManager::GetSDMpointer();
    for (G4LogicalVolume* lv : *lv_store)
    {
        // Look for the volume name
        CELER_ASSERT(lv);
        auto name_iter = sd_vol_names.find(lv->GetName());
        if (name_iter == sd_vol_names.end())
            continue;

        // Create SD, attach to volume, and save a reference to it
        auto sd = std::make_unique<SimpleSensitiveDetector>(lv);
        lv->SetSensitiveDetector(sd.get());
        sd_manager->AddNewDetector(sd.release());

        // Remove from the set of requested names
        sd_vol_names.erase(name_iter);
    }

    CELER_VALIDATE(sd_vol_names.empty(),
                   << "SD volumes were specified that don't exist in the "
                      "geometry: "
                   << join(sd_vol_names.begin(), sd_vol_names.end(), ", "));
    return result;
}

//---------------------------------------------------------------------------//
//! Restore SD map when rebuilding geometry
auto SensDetTestBase::build_geometry() -> SPConstCoreGeo
{
    // Build or fetch geo
    auto result = Base::build_geometry();

    for (G4LogicalVolume* lv : *G4LogicalVolumeStore::GetInstance())
    {
        // Add name, detector, volume to our lists
        if (auto* sd = lv->GetSensitiveDetector())
        {
            if (auto* ssd = dynamic_cast<SimpleSensitiveDetector*>(sd))
            {
                auto [iter, inserted] = detectors_.insert({lv->GetName(), ssd});
                CELER_VALIDATE(inserted,
                               << "duplicate sensitive detector name: "
                               << iter->first);

                // Clear hits if we're rebuilding the geometry
                ssd->clear();
            }
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
