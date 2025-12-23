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
#include "ReflectionFormSampler.hh"

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
 * equations with the \c FresnelCalculator and sampled to determine if the
 * photon will reflect or refract. If it reflects, then the UNIFIED model is
 * used to handle the different forms of reflection via \c
 * ReflectionFormSampler. If it refracts, then dielectric-dielectric interfaces
 * will use Snell's law to determine the refracted wave direction and
 * polarization. For dielectric-metal interfaces, refracted waves are just
 * absorbed.
 */
class DielectricInteractor
{
  public:
    struct Executor
    {
        NativeCRef<DielectricData> dielectric_data;
        NativeCRef<UnifiedReflectionData> unified_data;

        // Build and sample interactor for a track
        inline CELER_FUNCTION SurfaceInteraction
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
                         ReflectionModeSampler reflection_calc,
                         DielectricInterface dielectric_interface);

    // Sample the dielectric interaction
    template<class Engine>
    inline CELER_FUNCTION SurfaceInteraction operator()(Engine&) const;

  private:
    FresnelCalculator fresnel_;
    Real3 const& inc_direction_;
    ParticleTrackView const& inc_photon_;
    SurfacePhysicsTrackView const& surface_phys_;
    ReflectionModeSampler reflection_calc_;
    DielectricInterface dielectric_interface_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create an interactor and sample it for the given track.
 */
CELER_FUNCTION SurfaceInteraction
DielectricInteractor::Executor::operator()(CoreTrackView const& track) const
{
    auto s_phys = track.surface_physics();

    // Can't do analytic interaction if there's no post-volume optical material
    if (!s_phys.next_material())
    {
        return SurfaceInteraction::from_absorption();
    }

    // No interaction if optical materials are identical
    if (s_phys.material() == s_phys.next_material())
    {
        return SurfaceInteraction::from_transmission();
    }

    auto rng = track.rng();
    auto sub_model_id = s_phys.interface(SurfacePhysicsOrder::interaction)
                            .internal_surface_id();
    CELER_ASSERT(sub_model_id < dielectric_data.interface.size());

    return DielectricInteractor{
        track.particle(),
        track.geometry().dir(),
        s_phys,
        track.material_record(s_phys.material()),
        track.material_record(s_phys.next_material()),
        ReflectionModeSampler{
            unified_data, sub_model_id, track.particle().energy()},
        dielectric_data.interface[sub_model_id]}(rng);
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
    ReflectionModeSampler reflection_calc,
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
        return ReflectionFormSampler{
            reflection_calc_,
            ReflectionFormCalculator{
                inc_direction_, inc_photon_, surface_phys_}}(rng);
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
