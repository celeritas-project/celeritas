//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CeleritasG4.hh
//! \brief One-header include for Geant4 Celeritas applications
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if !CELERITAS_USE_GEANT4
#    error "Celeritas was not configured with Geant4 enabled"
#endif

#include "accel/AlongStepFactory.hh"
#include "accel/FastSimulationIntegration.hh"
#include "accel/FastSimulationModel.hh"
#include "accel/SetupOptions.hh"
#include "accel/TrackingManagerConstructor.hh"
#include "accel/TrackingManagerIntegration.hh"
#include "accel/UserActionIntegration.hh"
