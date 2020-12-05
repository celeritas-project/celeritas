//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Device.nocuda.cc
//---------------------------------------------------------------------------//
#include "Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * No device initialization takes place since CUDA is disabled.
 */
void initialize_device(const Communicator&) {}

//---------------------------------------------------------------------------//
/*!
 * CUDA is not enabled.
 */
bool is_device_enabled()
{
    return false;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
