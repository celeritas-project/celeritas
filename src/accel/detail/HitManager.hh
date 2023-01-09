//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "orange/Types.hh"
#include "celeritas/geo/GeoParamsFwd.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepInterface.hh"

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
 *   exclusions?)
 * - Maps those volumes to VecGeom geometry
 * - Creates a HitProcessor for each Geant4 thread
 *
 * Execute:
 * - Can share DetectorStepOutput across threads for now since StepGatherAction
 *   is mutexed across all threads
 * - Calls a single HitProcessor (thread safe because of caller's mutex)
 */
class HitManager final : public StepInterface
{
  public:
    // Construct with VecGeom for mapping volume IDs
    HitManager(const GeoParams& geo, const SDSetupOptions& setup);

    // Default destructor
    ~HitManager();

    // Selection of data required for this interface
    Filters filters() const final;

    // Selection of data required for this interface
    StepSelection selection() const final { return selection_; }

    // Process CPU-generated hits
    void execute(StateHostRef const&) final;

    // Process device-generated hits
    void execute(StateDeviceRef const&) final;

  private:
    bool                          nonzero_energy_deposition_{};
    StepSelection                 selection_;
    DetectorStepOutput            steps_;
    std::vector<VolumeId>         vecgeom_vols_;
    std::unique_ptr<HitProcessor> process_hits_;

    void call_local_processor() const;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
