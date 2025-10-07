//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "celeritas/optical/CoreTrackView.hh"

#include "FresnelCalculator.hh"
#include "UnifiedReflectionSampler.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical interactions for dielectric-dielectric and dielectric-metal
 * interfaces.
 *
 * For both interfaces, the reflectivity is first calculated from Fresnel
 * equations and sampled to determine if the photon will reflect or refract. If
 * it reflects, then the UNIFIED model is used to handle the different forms of
 * reflection. If it refracts, then dielectric-dielectric interfaces will use
 * Snell's law to determine the refracted wave direction and polarization. For
 * dielectric-metal interfaces, refracted waves are just absorbed.
 */
class DielectricInteractor
{
  public:
    struct Builder
    {
        NativeCRef<DielectricData> dielectric_data;
        NativeCRef<UnifiedReflectionData> unified_data;

        // Build the interactor for a track
        inline CELER_FUNCTION DielectricInteractor
        operator()(CoreTrackView const&) const;
    };

  public:
    // Construct interactor from track data
    inline CELER_FUNCTION
    DielectricInteractor(ParticleTrackView const& particle,
                         Real3 const& inc_direction,
                         SurfacePhysicsTrackView const& surface_physics,
                         MaterialView const& pre_material,
                         MaterialView const& post_material,
                         ReflectionFormCalculator reflection_calc,
                         DielectricInterface dielectric_interface);

    // Sample the dielectric interaction
    template<class Engine>
    inline CELER_FUNCTION SurfaceInteraction operator()(Engine&) const;

  private:
    FresnelCalculator fresnel_;
    Real3 const& inc_direction_;
    ParticleTrackView const& inc_photon_;
    SurfacePhysicsTrackView const& surface_phys_;
    ReflectionFormCalculator reflection_calc_;
    DielectricInterface dielectric_interface_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create an interactor for the given track.
 */
CELER_FUNCTION DielectricInteractor
DielectricInteractor::Builder::operator()(CoreTrackView const& track) const
{
    auto s_phys = track.surface_physics();
    auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::interaction)
                            .internal_surface_id();

    return DielectricInteractor{
        track.particle(),
        track.geometry().dir(),
        s_phys,
        track.material_record(s_phys.material()),
        track.material_record(s_phys.next_material()),
        ReflectionFormCalculator{
            unified_data, sub_model_id, track.particle().energy()},
        dielectric_data.interface[sub_model_id]};
}

//---------------------------------------------------------------------------//
/*!
 * Construct an interactor from track views.
 */
CELER_FUNCTION DielectricInteractor::DielectricInteractor(
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    SurfacePhysicsTrackView const& surface_physics,
    MaterialView const& pre_material,
    MaterialView const& post_material,
    ReflectionFormCalculator reflection_calc,
    DielectricInterface dielectric_interface)
    : fresnel_(inc_direction,
               particle,
               surface_physics.facet_normal(),
               pre_material,
               post_material)
    , inc_direction_(inc_direction)
    , inc_photon_(particle)
    , surface_phys_(surface_physics)
    , reflection_calc_(reflection_calc)
    , dielectric_interface_(dielectric_interface)
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the dielectric interaction.
 */
template<class Engine>
CELER_FUNCTION SurfaceInteraction
DielectricInteractor::operator()(Engine& rng) const
{
    if (BernoulliDistribution{fresnel_.calc_reflectivity()}(rng))
    {
        // Reflection
        return UnifiedReflectionSampler{
            reflection_calc_, inc_direction_, inc_photon_, surface_phys_}(rng);
    }
    else
    {
        // Refraction
        switch (dielectric_interface_)
        {
            case DielectricInterface::metal:
                return SurfaceInteraction::from_absorption();
            case DielectricInterface::dielectric:
                return fresnel_.refracted_interaction();
            default:
                CELER_ASSERT_UNREACHABLE();
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
