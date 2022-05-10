//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantVersion.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4Version.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Macro for differentiating between Geant4 v10 and v11.
 */
#if defined(G4VERSION_NUMBER) && G4VERSION_NUMBER < 1100
#    define CELERITAS_G4_V10 1
#else
#    define CELERITAS_G4_V10 0
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
