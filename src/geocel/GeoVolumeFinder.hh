//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoVolumeFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "GeoParamsInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Find a volume ID by searching for a label.
 *
 * This generally should be a fallback for looking directly from a Geant4
 * pointer. It will first do an exact match for the label, then do a fuzzier
 * search. It emits warnings for inexact matches and returns a null \c
 * ImplVolumeId if not found.
 */
class GeoVolumeFinder
{
  public:
    //!@{
    //! \name Type aliases
    using ImplVolumeMap = GeoParamsInterface::ImplVolumeMap;
    //!@}

  public:
    // Construct from an interface
    explicit inline GeoVolumeFinder(GeoParamsInterface const& geo);

    // Perform search
    ImplVolumeId operator()(Label const& label) const
        noexcept(!CELERITAS_DEBUG);

  private:
    ImplVolumeMap const& vols_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from geometry volume names.
 */
GeoVolumeFinder::GeoVolumeFinder(GeoParamsInterface const& geo)
    : vols_{geo.impl_volumes()}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
