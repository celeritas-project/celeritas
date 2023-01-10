//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/Logger.hh
//! \brief Geant4-friendly logging utilities
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/io/Logger.hh"

class G4RunManager;

namespace celeritas
{
//---------------------------------------------------------------------------//
// Create a multithread-friendly logger
Logger make_mt_logger(G4RunManager const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
