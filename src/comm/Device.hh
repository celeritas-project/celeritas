//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Device.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Communicator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Whether device code should be used
bool is_device_enabled();

//---------------------------------------------------------------------------//
// Initialize device in a round-robin fashion from a communicator
void initialize_device(const Communicator& comm);

//---------------------------------------------------------------------------//
} // namespace celeritas
