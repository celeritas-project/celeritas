//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "orange/Types.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/user/StepInterface.hh"

class G4LogicalVolume;

namespace celeritas
{
struct SDSetupOptions;

namespace detail
{
class HitProcessor;

//---------------------------------------------------------------------------//
/*!
 * Manage the conversion of hits from Celeritas to Geant4.
 *
 * Construction:
 * - Created during SharedParams::Initialize alongside the step collector
 * - Is shared across threads
 * - Finds *all* logical volumes that have SDs attached (TODO: add list of
 *   exclusions for SDs that are implemented natively on GPU)
 * - Maps those volumes to VecGeom geometry
 * - Creates a HitProcessor for each Geant4 thread
 *
 * \warning Because of low-level problems with Geant4 allocators, the
 */
class HitManager final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using StepStateHostRef = HostRef<StepStateData>;
    using StepStateDeviceRef = DeviceRef<StepStateData>;
    using SPConstVecLV
        = std::shared_ptr<const std::vector<G4LogicalVolume const*>>;
    using VecVolId = std::vector<VolumeId>;
    //!@}

  public:
    // Construct with VecGeom for mapping volume IDs
    HitManager(GeoParams const& geo,
               SDSetupOptions const& setup,
               StreamId::size_type num_streams);

    // Default destructor
    ~HitManager();

    // Selection of data required for this interface
    Filters filters() const final;

    // Selection of data required for this interface
    StepSelection selection() const final { return selection_; }

    // Process CPU-generated hits
    void process_steps(HostStepState) final;

    // Process device-generated hits
    void process_steps(DeviceStepState) final;

    // Destroy local data to avoid Geant4 crashes
    void finalize(StreamId sid);

    //// ACCESSORS ////

    //! Access the logical volumes that have SDs attached
    SPConstVecLV const& geant_vols() const { return geant_vols_; }

    //! Access the Celeritas volume IDs corresponding to the detectors
    VecVolId const& celer_vols() const { return vecgeom_vols_; }

  private:
    using VecLV = std::vector<G4LogicalVolume const*>;

    bool nonzero_energy_deposition_{};
    bool locate_touchable_{};
    StepSelection selection_;
    SPConstVecLV geant_vols_;
    VecVolId vecgeom_vols_;
    std::vector<std::unique_ptr<HitProcessor>> processors_;

    // Ensure thread-local hit processor exists and return it
    HitProcessor& get_local_hit_processor(StreamId);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
