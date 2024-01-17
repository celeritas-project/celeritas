//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
// Manually create a multithread-friendly logger
Logger MakeMTLogger(G4RunManager const&);

//---------------------------------------------------------------------------//
//! Manually create a multithread-friendly logger (remove in v1.0)
[[deprecated]] inline Logger make_mt_logger(G4RunManager const& rm)
{
    return MakeMTLogger(rm);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
