//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoVolumeView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "geocel/Types.hh"

#include "VolumeData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access material and connectivity for a single logical volume.
 *
 * This GPU-compatible view provides per-volume access to the geometry
 * material ID and parent/child edges in the volume instance graph.
 *
 * \code
   GeoVolumeView view{params.host_ref(), vol_id};
   GeoMatId mat = view.material();
   for (VolumeInstanceId vi : view.children()) { ... }
 * \endcode
 */
class GeoVolumeView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<VolumeParamsData>;
    using SpanVolInst = Span<VolumeInstanceId const>;
    //!@}

  public:
    // Construct with shared data and a volume ID
    explicit inline CELER_FUNCTION
    GeoVolumeView(ParamsRef const& params, VolumeId vol_id);

    //! Volume being viewed
    CELER_FUNCTION VolumeId volume_id() const { return vol_id_; }

    // Get the geometry material ID for this volume
    inline CELER_FUNCTION GeoMatId material() const;

    // Get the incoming edges (volume instances that place this volume)
    inline CELER_FUNCTION SpanVolInst parents() const;

    // Get the outgoing edges (child instances placed inside this volume)
    inline CELER_FUNCTION SpanVolInst children() const;

  private:
    ParamsRef const& params_;
    VolumeId vol_id_;

    CELER_FORCEINLINE_FUNCTION VolumeRecord const& record() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared data and a volume ID.
 */
CELER_FUNCTION
GeoVolumeView::GeoVolumeView(ParamsRef const& params, VolumeId vol_id)
    : params_(params), vol_id_(vol_id)
{
    CELER_EXPECT(vol_id_ < params_.volumes.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get the geometry material ID for this volume.
 */
CELER_FUNCTION GeoMatId GeoVolumeView::material() const
{
    return this->record().material;
}

//---------------------------------------------------------------------------//
/*!
 * Get the incoming edges (volume instances that place this volume).
 */
CELER_FUNCTION auto GeoVolumeView::parents() const -> SpanVolInst
{
    return params_.vi_storage[this->record().parents];
}

//---------------------------------------------------------------------------//
/*!
 * Get the outgoing edges (child instances placed inside this volume).
 */
CELER_FUNCTION auto GeoVolumeView::children() const -> SpanVolInst
{
    return params_.vi_storage[this->record().children];
}

//---------------------------------------------------------------------------//
CELER_FUNCTION VolumeRecord const& GeoVolumeView::record() const
{
    return params_.volumes[vol_id_];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
