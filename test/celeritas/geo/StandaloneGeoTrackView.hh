//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/StandaloneGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "celeritas/geo/CoreGeoTrackView.hh"

#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_DYNAMIC
#    include "geocel/WrappedGeoTrackView.hh"
#    include "orange/OrangeTrackView.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Concrete core geometry track view constructed from geometry data.
 *
 * This is the core geometry track view, except for the \c dynamic core
 * geometry: there, \c CoreGeoTrackView is an abstract interface owned by the
 * core state, so tests that use only geometry params and state instead wrap
 * the underlying implementation in the same virtual interface.
 */
#if CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_DYNAMIC
using StandaloneGeoTrackView = WrappedGeoTrackView<OrangeTrackView>;
#else
using StandaloneGeoTrackView = CoreGeoTrackView;
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
