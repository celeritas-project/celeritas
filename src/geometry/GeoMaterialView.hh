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
    // Construct for the given particle and material ids
    inline CELER_FUNCTION
    GeoMaterialView(const GeoMaterialPointers& params, VolumeId volume);

    //! Return material
    CELER_FORCEINLINE_FUNCTION MaterialId material() const { return mat_; }

  private:
    MaterialId mat_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from shared data and current volume.
 *
 * Note that this will *fail* if the particle is outside -- the volume ID will
 * be false.
 */
CELER_FUNCTION
GeoMaterialView::GeoMaterialView(const GeoMaterialPointers& params,
                                 VolumeId                   volume)
{
    CELER_EXPECT(volume < params.materials.size());
    mat_ = params.materials[volume];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
