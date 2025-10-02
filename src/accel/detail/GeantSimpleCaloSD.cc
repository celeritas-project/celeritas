//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/GeantSimpleCaloSD.cc
//---------------------------------------------------------------------------//
#include "GeantSimpleCaloSD.hh"

#include "corecel/Assert.hh"
#include "geocel/GeantGeoUtils.hh"

#include "GeantSimpleCaloStorage.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with name and shared storage.
 */
GeantSimpleCaloSD::GeantSimpleCaloSD(SPStorage storage, size_type thread_id)
    : G4VSensitiveDetector{storage->name}
    , storage_{std::move(storage)}
    , thread_id_{thread_id}
{
    CELER_EXPECT(storage_);
    CELER_EXPECT(thread_id < storage_->data.size());
    CELER_EXPECT(storage_->data[thread_id].size()
                 == storage_->volume_to_index.size());
}

//---------------------------------------------------------------------------//
/*!
 * Add energy deposition from this step to the corresponding logical volume.
 */
bool GeantSimpleCaloSD::ProcessHits(G4Step* g4step, G4TouchableHistory*)
{
    CELER_EXPECT(g4step);
    CELER_EXPECT(g4step->GetPreStepPoint());
    double const edep = g4step->GetTotalEnergyDeposit();

    if (edep == 0)
    {
        return false;
    }

    auto const* pv = g4step->GetPreStepPoint()->GetPhysicalVolume();
    CELER_ASSERT(pv);
    auto det_id_iter = storage_->volume_to_index.find(pv->GetLogicalVolume());
    CELER_VALIDATE(det_id_iter != storage_->volume_to_index.end(),
                   << "logical volume " << StreamableLV{pv->GetLogicalVolume()}
                   << " is not attached to simple calo '" << storage_->name
                   << "'";);

    auto& thread_data = storage_->data[thread_id_];
    CELER_ASSERT(det_id_iter->second < thread_data.size());
    thread_data[det_id_iter->second] += edep;
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
