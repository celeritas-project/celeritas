//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterialView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeoMaterialData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access geometry-to-material conversion.
 */
class GeoMaterialView
{
  public:
    //!@{
    //! \name Type aliases
    using GeoMaterialData = NativeCRef<GeoMaterialParamsData>;
    //!@}

  public:
    // Construct from shared data
    inline CELER_FUNCTION GeoMaterialView(GeoMaterialData const& params);

    // Return material for the given volume
    inline CELER_FUNCTION MaterialId material_id(VolumeId volume) const;

  private:
    GeoMaterialData const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared data.
 */
CELER_FUNCTION
GeoMaterialView::GeoMaterialView(GeoMaterialData const& params)
    : params_(params)
{
}

//---------------------------------------------------------------------------//
/*!
 * Return material for the given volume.
 *
 * Note that this will *fail* if the particle is outside -- the volume ID will
 * be false.
 */
CELER_FUNCTION MaterialId GeoMaterialView::material_id(VolumeId volume) const
{
    CELER_EXPECT(volume < params_.materials.size());
    return params_.materials[volume];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
