//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
template<Ownership W, MemSpace M>
struct DielectricData
{
    //!@{
    //! \name Type aliases
    template<class T>
    SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    SurfaceItems<bool> is_metal;
};

//---------------------------------------------------------------------------//
/*!
 */
class DielectricInteractor
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

    struct Builder
    {
        NativeCRef<DielectricData> dielectric_data;
        NativeCRef<UnifiedReflectionData> unified_data;

        inline CELER_FUNCTION DielectricInteractor
        operator()(CoreTrackView const&) const;
    };

  public:
    inline CELER_FUNCTION
    DielectricInteractor(ParticleTrackView const& particle,
                         Real3 const& inc_direction,
                         SurfacePhysicsTrackView const& surface_physics,
                         MaterialView const& pre_material,
                         MaterialView const& post_material);

    template<class Engine>
    inline CELER_FUNCTION SurfaceInteraction operator()(Engine&) const;

  private:
    FresnelCalculator fresnel_;
    Real3 const& inc_direction_;
    ParticleTrackView const& inc_photon_;
    SurfacePhysicsTrackView const& surface_phys_;
    UnifiedReflectionView unified_reflection_;
    bool is_metal_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
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
        track.material(s_phys.material()),
        track.material(s_phys.next_material()),
        UnifiedReflectionView{unified_data, sub_model_id},
        dielectric_data.is_metal[sub_model_id]};
}

CELER_FUNCTION DielectricInteractor::DielectricInteractor(
    ParticleTrackView const& particle,
    Real3 const& inc_direction,
    SurfacePhysicsTrackView const& surface_physics,
    MaterialView const& pre_material,
    MaterialView const& post_material,
    UnifiedReflectionView const& unified_reflection,
    bool is_metal)
    : fresnel_(inc_direction,
               particle,
               surface_physics.facet_normal(),
               pre_material,
               post_material)
    , inc_direction_(inc_direction)
    , inc_photon_(particle)
    , surface_phys_(surface_physics)
    , unified_reflection_(unified_reflection)
    , is_metal_(is_metal)
{
}

template<class Engine>
CELER_FUNCTION SurfaceInteraction
DielectricInteraction::operator()(Engine& rng) const
{
    if (BernoulliDistribution{fresnel_.calc_reflectivity()}(rng))
    {
        // Reflection
        return UnifiedReflectionSampler{
            unified_reflection_, inc_photon_, inc_direction_, surface_phys_}(
            rng);
    }
    else
    {
        // Refraction
        if (is_metal_)
        {
            return SurfaceInteraction::from_absorbed();
        }
        else
        {
            return fresnel_.refracted_interaction();
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
