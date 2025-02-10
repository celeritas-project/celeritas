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
// Manually create a multithread-friendly logger
Logger MakeMTLogger(G4RunManager const&);

//---------------------------------------------------------------------------//
//! Manually create a multithread-friendly logger (remove in v0.6)
[[deprecated]] inline Logger make_mt_logger(G4RunManager const& rm)
{
    return MakeMTLogger(rm);
}

//---------------------------------------------------------------------------//
// Get the thread ID printed to logger messages.
std::string get_thread_label();

//---------------------------------------------------------------------------//
}  // namespace celeritas
