//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterialView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeoMaterialInterface.hh"

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
    //! Type aliases
    using GeoMaterialPointers
        = GeoMaterialParamsData<Ownership::const_reference, MemSpace::native>;
    //!@}

  public:
    // Construct from shared data
    inline CELER_FUNCTION GeoMaterialView(const GeoMaterialPointers& params);

    // Return material for the given volume
    inline CELER_FUNCTION MaterialId material_id(VolumeId volume) const;

  private:
    const GeoMaterialPointers& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared data.
 */
CELER_FUNCTION
GeoMaterialView::GeoMaterialView(const GeoMaterialPointers& params)
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
} // namespace celeritas
