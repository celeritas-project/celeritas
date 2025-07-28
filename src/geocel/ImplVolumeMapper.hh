//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ImplVolumeMapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map a the implementation volume ID of a specific geometry into Celeritas
 * volume IDs.
 *
 * The implementation volume IDs are details of the specific geometry's
 * implementation of the detector geometry. The Celeritas volume IDs correspond
 * to the common volume DAG described by \c VolumeParams . This class
 * encapsulates the mapping between a specific geometry implementation's
 * volume IDs and the common volume IDs.
 */
class ImplVolumeMapper
{
  public:
    inline CELER_FUNCTION ImplVolumeId operator()(VolumeId) const;
    inline CELER_FUNCTION VolumeId operator()(ImplVolumeId) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION ImplVolumeId ImplVolumeMapper::operator()(VolumeId v) const
{
    return ImplVolumeId(v.unchecked_get());
}

CELER_FUNCTION VolumeId ImplVolumeMapper::operator()(ImplVolumeId v) const
{
    return VolumeId(v.unchecked_get());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
