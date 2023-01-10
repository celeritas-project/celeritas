//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantConfig.hh
//! \brief Define macros for Geant4 versions and features.
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

/*!
 * \def CELERITAS_G4_V10
 *
 * Macro constant for differentiating between Geant4 v10 and v11.
 */
#if CELERITAS_USE_GEANT4 || defined(__DOXYGEN__)

#    include <G4Version.hh>
#    if defined(G4VERSION_NUMBER) && G4VERSION_NUMBER < 1100
#        define CELERITAS_G4_V10 1
#    else
#        define CELERITAS_G4_V10 0
#    endif

#    include <G4GlobalConfig.hh>
#    ifdef G4MULTITHREADED
#        define CELERITAS_G4_MT 1
#    else
#        define CELERITAS_G4_MT 0
#    endif
#endif
