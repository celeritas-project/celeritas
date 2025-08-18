//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSd.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/cont/EnumArray.hh"
#include "geocel/Types.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/user/StepInterface.hh"

class G4LogicalVolume;
class G4ParticleDefinition;

namespace celeritas
{
//---------------------------------------------------------------------------//
namespace inp
{
struct GeantSd;
}
namespace detail
{
class HitProcessor;
}

class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Hit Geant4 sensitive detectors with Celeritas steps.
 *
 * Construction:
 * - Created during SharedParams::Initialize alongside the step collector
 * - Is shared across threads
 * - Finds all logical volumes that have SDs attached
 * - Maps those volumes to Celeritas geometry
 *
 * Because of low-level use of Geant4 allocators through the associated Geant4
 * objects, the hit processors \em must be allocated and deallocated on the
 * same thread in which they're used, so \c make_local_processor is deferred
 * until after construction and called in the \c LocalTransporter constructor.
 */
class GeantSd final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using StepStateHostRef = HostRef<StepStateData>;
    using StepStateDeviceRef = DeviceRef<StepStateData>;
    using SPConstVecLV
        = std::shared_ptr<std::vector<G4LogicalVolume const*> const>;
    using HitProcessor = detail::HitProcessor;
    using SPProcessor = std::shared_ptr<HitProcessor>;
    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using VecVolId = std::vector<ImplVolumeId>;
    using VecParticle = std::vector<G4ParticleDefinition const*>;
    using StepPointBool = EnumArray<StepPoint, bool>;
    using Input = inp::GeantSd;
    //!@}

  public:
    // Construct with Celeritas objects for mapping
    GeantSd(SPConstCoreGeo geo,
            ParticleParams const& par,
            Input const& setup,
            StreamId::size_type num_streams);

    CELER_DEFAULT_MOVE_DELETE_COPY(GeantSd);

    // Default destructor
    ~GeantSd();

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
    StepPointBool const& locate_touchable() const { return locate_touchable_; }

  private:
    using VecLV = std::vector<G4LogicalVolume const*>;

    bool nonzero_energy_deposition_{};
    VecVolId celer_vols_;

    // Hit processor setup
    SPConstCoreGeo geo_;
    SPConstVecLV geant_vols_;
    VecParticle particles_;
    StepSelection selection_;
    StepPointBool locate_touchable_{};

    std::vector<std::weak_ptr<HitProcessor>> processor_weakptrs_;
    std::vector<HitProcessor*> processors_;

    // Construct vecgeom/geant volumes
    void setup_volumes(CoreGeoParams const& geo, Input const& setup);
    // Construct celeritas/geant particles
    void setup_particles(ParticleParams const& par);

    // Ensure thread-local hit processor exists and return it
    HitProcessor& get_local_hit_processor(StreamId);
};

#if !CELERITAS_USE_GEANT4

inline GeantSd::GeantSd(SPConstCoreGeo,
                        ParticleParams const&,
                        Input const&,
                        StreamId::size_type)
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline GeantSd::~GeantSd() = default;

inline GeantSd::SPProcessor GeantSd::make_local_processor(StreamId)
{
    CELER_ASSERT_UNREACHABLE();
}

inline GeantSd::Filters GeantSd::filters() const
{
    CELER_ASSERT_UNREACHABLE();
}

inline void GeantSd::process_steps(HostStepState)
{
    CELER_ASSERT_UNREACHABLE();
}

inline void GeantSd::process_steps(DeviceStepState)
{
    CELER_ASSERT_UNREACHABLE();
}

#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
