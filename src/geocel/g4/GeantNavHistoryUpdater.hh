//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantNavHistoryUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "geocel/Types.hh"

class G4NavigationHistory;

namespace celeritas
{
class GeantGeoParams;
namespace detail
{
class GeantVolumeInstanceMapper;
}

//---------------------------------------------------------------------------//
/*!
 * Update a nav history to match the given volume instance stack.
 *
 * This requires metadata from \c celeritas::global_geant_geo .
 *
 * The constructed nav history always has at least one level (i.e. GetDepth is
 * zero). An empty input stack, corresponding to "outside" the world, results
 * in a nav history with one level but a \c nullptr physical volume as the top.
 *
 * \note The stack should have the same semantics as \c VolumeLevelId, i.e. the
 * initial entry is the "most global" level.
 */
class GeantNavHistoryUpdater
{
  public:
    //!@{
    //! \name Type aliases
    using VIMapper = detail::GeantVolumeInstanceMapper;
    //!@}

  public:
    // Construct using global geant geo
    explicit GeantNavHistoryUpdater(GeantGeoParams const&);
    //! Construct with volume instance mapper
    explicit GeantNavHistoryUpdater(VIMapper const& mapper) : mapper_{mapper}
    {
    }

    // Reconstruct into the given nav history
    void
    operator()(Span<VolumeInstanceId const> stack, G4NavigationHistory* nav);

  private:
    VIMapper const& mapper_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
