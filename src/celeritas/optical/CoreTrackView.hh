//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CoreTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/engine/RngEngine.hh"
#include "geocel/VolumeSurfaceView.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"

#include "CoreTrackData.hh"
#include "MaterialView.hh"
#include "ParticleTrackView.hh"
#include "PhysicsTrackView.hh"
#include "SimTrackView.hh"
#include "TrackInitializer.hh"
#include "surface/SurfacePhysicsTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#endif

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Access all core properties of a physics track.
 */
class CoreTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<CoreParamsData>;
    using StateRef = NativeRef<CoreStateData>;
    //!@}

  public:
    // Construct directly from a track slot ID
    inline CELER_FUNCTION CoreTrackView(ParamsRef const& params,
                                        StateRef const& states,
                                        TrackSlotId slot);

    // Initialize the track states
    inline CELER_FUNCTION CoreTrackView& operator=(TrackInitializer const&);

    // Return a geometry view
    inline CELER_FUNCTION GeoTrackView geometry() const;

    // Return a material view
    inline CELER_FUNCTION MaterialView material_record() const;

    // Return a material view (using an existing geo view)
    inline CELER_FUNCTION MaterialView material_record(GeoTrackView const&) const;

    // Return a material view for a specific optical material
    inline CELER_FUNCTION MaterialView material_record(OptMatId) const;

    // Return a simulation management view
    inline CELER_FUNCTION SimTrackView sim() const;

    // Return a particle view
    inline CELER_FUNCTION ParticleTrackView particle() const;

    // Return a physics view
    inline CELER_FUNCTION PhysicsTrackView physics() const;

    // Return a volume surface view
    inline CELER_FUNCTION VolumeSurfaceView surface() const;

    // Return a volume surface view from volume ID
    inline CELER_FUNCTION VolumeSurfaceView surface(VolumeId) const;

    // Return a surface physics view
    inline CELER_FUNCTION SurfacePhysicsTrackView surface_physics() const;

    // Return an RNG engine
    inline CELER_FUNCTION RngEngine rng() const;

    // Get the track's index among the states
    inline CELER_FUNCTION TrackSlotId track_slot_id() const;

    // Flag a track for deletion
    inline CELER_FUNCTION void apply_errored();

  private:
    ParamsRef const& params_;
    StateRef const& states_;
    TrackSlotId const track_slot_id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with comprehensive param/state data and track slot.
 *
 * For optical tracks, the value of the track slot is the same as the track ID.
 */
CELER_FUNCTION
CoreTrackView::CoreTrackView(ParamsRef const& params,
                             StateRef const& states,
                             TrackSlotId track_slot)
    : params_(params), states_(states), track_slot_id_(track_slot)
{
    CELER_EXPECT(track_slot_id_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 */
CELER_FUNCTION CoreTrackView&
CoreTrackView::operator=(TrackInitializer const& init)
{
    // Initialiize the sim state
    this->sim() = SimTrackView::Initializer{init.time};

    // Initialize the geometry state
    auto geo = this->geometry();
    geo = GeoTrackInitializer{init.position, init.direction, {}};
    if (CELER_UNLIKELY(geo.failed() || geo.is_outside()))
    {
#if !CELER_DEVICE_COMPILE
        if (geo.is_outside())
        {
            // Print an error message if initialization was "successful" but
            // track is outside
            CELER_LOG_LOCAL(error) << "Track started outside the geometry";
        }
#endif
        this->apply_errored();
        return *this;
    }

    // Initialize the particle state
    this->particle()
        = ParticleTrackView::Initializer{init.energy, init.polarization};

    // Initialize the physics state
    this->physics() = PhysicsTrackView::Initializer{};

    // Initialize the surface state
    this->surface_physics().reset();

    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Return a geometry view.
 */
CELER_FUNCTION auto CoreTrackView::geometry() const -> GeoTrackView
{
    return GeoTrackView{
        params_.geometry, states_.geometry, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a material view.
 */
CELER_FORCEINLINE_FUNCTION auto CoreTrackView::material_record() const
    -> MaterialView
{
    return this->material_record(this->geometry());
}

//---------------------------------------------------------------------------//
/*!
 * Return a material view using an existing geo track view.
 */
CELER_FUNCTION auto
CoreTrackView::material_record(GeoTrackView const& geo) const -> MaterialView
{
    CELER_EXPECT(!geo.is_outside());
    return MaterialView{params_.material, geo.impl_volume_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a material view for a specific optical material ID.
 */
CELER_FUNCTION auto CoreTrackView::material_record(OptMatId opt_mat) const
    -> MaterialView
{
    auto material = MaterialView{params_.material, opt_mat};
    CELER_ENSURE(material);
    return material;
}

//---------------------------------------------------------------------------//
/*!
 * Return a particle view.
 */
CELER_FUNCTION auto CoreTrackView::particle() const -> ParticleTrackView
{
    return ParticleTrackView{states_.particle, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a physics view.
 */
CELER_FUNCTION auto CoreTrackView::physics() const -> PhysicsTrackView
{
    OptMatId mat_id = this->material_record().material_id();
    CELER_ASSERT(mat_id);
    return PhysicsTrackView{
        params_.physics, states_.physics, mat_id, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a volume surface view into the track's current volume.
 */
CELER_FUNCTION auto CoreTrackView::surface() const -> VolumeSurfaceView
{
    return this->surface(this->geometry().volume_id());
}

//---------------------------------------------------------------------------//
/*!
 * Return a volume surface view from volume ID.
 */
CELER_FUNCTION auto CoreTrackView::surface(VolumeId vol) const
    -> VolumeSurfaceView
{
    CELER_EXPECT(vol);
    return VolumeSurfaceView{params_.surface, vol};
}

//---------------------------------------------------------------------------//
/*!
 * Return a surface physics view.
 */
CELER_FUNCTION auto CoreTrackView::surface_physics() const
    -> SurfacePhysicsTrackView
{
    return SurfacePhysicsTrackView{params_.surface_physics,
                                   states_.surface_physics,
                                   this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return the RNG engine.
 */
CELER_FUNCTION auto CoreTrackView::rng() const -> RngEngine
{
    return RngEngine{params_.rng, states_.rng, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Return a simulation management view.
 */
CELER_FUNCTION SimTrackView CoreTrackView::sim() const
{
    return SimTrackView{params_.sim, states_.sim, this->track_slot_id()};
}

//---------------------------------------------------------------------------//
/*!
 * Get the track's index among the states.
 */
CELER_FORCEINLINE_FUNCTION TrackSlotId CoreTrackView::track_slot_id() const
{
    return track_slot_id_;
}

//---------------------------------------------------------------------------//
/*!
 * Set the 'errored' flag and tracking cut post-step action.
 *
 * \pre This cannot be applied if the current action is *after* post-step. (You
 * can't guarantee for example that sensitive detectors will pick up the energy
 * deposition.)
 *
 * \todo Add a tracking cut action? Currently the track is simply killed.
 */
CELER_FUNCTION void CoreTrackView::apply_errored()
{
    auto sim = this->sim();
    CELER_EXPECT(is_track_valid(sim.status()));
    sim.status(TrackStatus::errored);
    sim.post_step_action(params_.scalars.tracking_cut_action);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
