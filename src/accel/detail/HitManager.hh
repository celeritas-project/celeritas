//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
//---------------------------------------------------------------------------//

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
 *
 * Because of low-level use of Geant4 allocators through the associated Geant4
 * objects, the hit processors \em must be allocated and deallocated on the
 * same thread in which they're used, so \c make_local_processor is deferred
 * until after construction and called in the \c LocalTransporter constructor.
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
    using SPProcessor = std::shared_ptr<HitProcessor>;
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    using VecVolId = std::vector<VolumeId>;
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    //!@}

  public:
    // Construct with Celeritas objects for mapping
    HitManager(SPConstGeo geo,
               ParticleParams const& par,
               SDSetupOptions const& setup,
               StreamId::size_type num_streams);

    CELER_DEFAULT_MOVE_DELETE_COPY(HitManager);

    // Default destructor
    ~HitManager();

    // Create local hit processor
    SPProcessor make_local_processor(StreamId sid);

    // Selection of data required for this interface
    Filters filters() const final;

    // Selection of data required for this interface
    StepSelection selection() const final { return selection_; }

    // Process CPU-generated hits
    void process_steps(HostStepState) final;

    // Process device-generated hits
    void process_steps(DeviceStepState) final;

    //// ACCESSORS ////

    //! Access the logical volumes that have SDs attached
    SPConstVecLV const& geant_vols() const { return geant_vols_; }

    //! Access the Celeritas volume IDs corresponding to the detectors
    VecVolId const& celer_vols() const { return celer_vols_; }

    //! Access mapped particles if recreating G4Tracks later
    VecParticle const& geant_particles() const { return particles_; }

    //! Whether detailed volume information is reconstructed
    bool locate_touchable() const { return locate_touchable_; }

  private:
    using VecLV = std::vector<G4LogicalVolume const*>;

    bool nonzero_energy_deposition_{};
    VecVolId celer_vols_;

    // Hit processor setup
    SPConstGeo geo_;
    SPConstVecLV geant_vols_;
    VecParticle particles_;
    StepSelection selection_;
    bool locate_touchable_{};

    std::vector<std::weak_ptr<HitProcessor>> processor_weakptrs_;
    std::vector<HitProcessor*> processors_;

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
