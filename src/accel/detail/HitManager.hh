//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "geocel/Types.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/user/StepInterface.hh"

class G4LogicalVolume;
class G4ParticleDefinition;

namespace celeritas
{
struct SDSetupOptions;
class ParticleParams;

namespace detail
{
class HitProcessor;
//---------------------------------------------------------------------------//

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
 * \warning Because of low-level problems with Geant4 allocators, the hit
 * processors must be allocated and deallocated on the same thread in which
 * they're used.
 */
class HitManager final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using StepStateHostRef = HostRef<StepStateData>;
    using StepStateDeviceRef = DeviceRef<StepStateData>;
    using SPConstVecLV
        = std::shared_ptr<std::vector<G4LogicalVolume const*> const>;
    using VecVolId = std::vector<VolumeId>;
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    //!@}

  public:
    // Construct with Celeritas objects for mapping
    HitManager(GeoParams const& geo,
               ParticleParams const& par,
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

    //! Access mapped particles if recreating G4Tracks later
    VecParticle const& geant_particles() const { return particles_; }

  private:
    using VecLV = std::vector<G4LogicalVolume const*>;

    bool nonzero_energy_deposition_{};
    VecVolId vecgeom_vols_;

    // Hit processor setup
    SPConstVecLV geant_vols_;
    VecParticle particles_;
    StepSelection selection_;
    bool locate_touchable_{};

    std::vector<std::unique_ptr<HitProcessor>> processors_;

    // Construct vecgeom/geant volumes
    void setup_volumes(GeoParams const& geo, SDSetupOptions const& setup);
    // Construct celeritas/geant particles
    void setup_particles(ParticleParams const& par);

    // Ensure thread-local hit processor exists and return it
    HitProcessor& get_local_hit_processor(StreamId);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
