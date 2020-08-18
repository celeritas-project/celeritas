//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file HostValue.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Type-safe wrapper to indicate CPU/host storage.
 *
 * This should be used primarily to wrap \c Pointers classes returned from host
 * utility code (e.g. from a \c Store class).
 *
 * \sa DeviceValue
 */
template<class T>
struct HostValue
{
    T value;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
