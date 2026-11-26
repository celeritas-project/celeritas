//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockView.hh
//! \sa celeritas/woodcock/Woodcock.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "geocel/AllVolumesView.hh"
#include "celeritas/grid/UniformLogGridCalculator.hh"

#include "WoodcockData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access Woodcock tracking data.
 *
 * The view can be constructed and queried for applicability even if Woodcock
 * tracking is disabled.
 */
class WoodcockView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<WoodcockParamsData>;
    using Energy = WoodcockScalars::Energy;
    //!@}

    //! Enclosing tracking volume and its level in the volume path
    struct Root
    {
        WoodcockRootId id;
        VolumeLevelId level;

        //! Whether the track is in a root
        explicit CELER_FUNCTION operator bool() const
        {
            return static_cast<bool>(id);
        }
    };

  public:
    // Construct from shared data
    explicit CELER_FORCEINLINE_FUNCTION WoodcockView(ParamsRef const& params);

    // Whether the particle can potentially use Woodcock tracking
    inline CELER_FUNCTION bool is_applicable(ParticleId particle,
                                             Energy energy) const;

    // Whether a volume can be inside of a root
    inline CELER_FUNCTION bool can_be_inside(VolumeId volume) const;

    // Find the outermost root volume in the track's path
    template<class GTV>
    inline CELER_FUNCTION Root find_root(GTV const& geo,
                                         AllVolumesView const& volumes) const;

    // Calculate the majorant cross section
    inline CELER_FUNCTION real_type calc_majorant(WoodcockRootId root,
                                                  Energy energy) const;

    //! Maximum number of flights per step
    CELER_FUNCTION size_type max_flights() const
    {
        return params_.scalars.max_flights;
    }

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared data.
 */
CELER_FUNCTION WoodcockView::WoodcockView(ParamsRef const& params)
    : params_(params)
{
}

//---------------------------------------------------------------------------//
/*!
 * Whether the particle can potentially use Woodcock tracking.
 *
 * This is always false when Woodcock tracking is disabled.
 */
CELER_FUNCTION bool WoodcockView::is_applicable(ParticleId particle,
                                                Energy energy) const
{
    CELER_EXPECT(particle);
    return particle == params_.scalars.gamma
           && energy >= params_.scalars.min_energy
           && energy <= params_.scalars.max_energy;
}

//---------------------------------------------------------------------------//
/*!
 * Whether a volume can be inside of a root.
 *
 * The volume can be placed both inside and outside of a root, in which case \c
 * find_root is needed to determine whether the track really is inside.
 */
CELER_FUNCTION bool WoodcockView::can_be_inside(VolumeId volume) const
{
    CELER_EXPECT(params_);
    return volume && params_.inside[volume];
}

//---------------------------------------------------------------------------//
/*!
 * Find the outermost root volume in the track's path.
 *
 * The result is null if the track is not inside a root.
 */
template<class GTV>
CELER_FUNCTION auto WoodcockView::find_root(
    GTV const& geo, AllVolumesView const& volumes) const -> Root
{
    CELER_EXPECT(params_);

    Root result;
    geo.foreach_volume_path([&result, &volumes, &roots = params_.roots](
                                VolumeLevelId level, VolumeInstanceId vi) {
        VolumeId const vol = vi ? volumes.volume_id(vi) : volumes.world();
        if (WoodcockRootId id = roots[vol];
            id && (!result || level < result.level))
        {
            result = {id, level};
        }
    });
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the majorant cross section.
 */
CELER_FUNCTION real_type WoodcockView::calc_majorant(WoodcockRootId root,
                                                     Energy energy) const
{
    CELER_EXPECT(root < params_.majorants.size());
    return UniformLogGridCalculator{params_.majorants[root],
                                    params_.reals}(energy);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
