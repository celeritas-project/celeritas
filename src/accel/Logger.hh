//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
// Manually create a G4MT-friendly logger for event-specific info
Logger MakeMTSelfLogger(G4RunManager const&);

//---------------------------------------------------------------------------//
// Manually create a logger for setup info
Logger MakeMTWorldLogger(G4RunManager const&);

//---------------------------------------------------------------------------//
//! Manually create a multithread-friendly logger
//! \deprecated Remove in v1.0; replaced by MakeMTSelfLogger
[[deprecated]] inline Logger MakeMTLogger(G4RunManager const& rm)
{
    return MakeMTSelfLogger(rm);
}

//---------------------------------------------------------------------------//
// Get the thread ID printed to logger messages.
std::string get_thread_label();

//---------------------------------------------------------------------------//
}  // namespace celeritas
