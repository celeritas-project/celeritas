//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CollectionMirror.i.hh
//---------------------------------------------------------------------------//
#include <utility>
#include "comm/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<template<Ownership, MemSpace> class P>
CollectionMirror<P>::CollectionMirror(HostValue&& host)
    : host_(std::move(host))
{
    CELER_EXPECT(host_);
    host_ref_ = host_;
    if (celeritas::device())
    {
        // Copy data to device and save reference
        device_     = host_;
        device_ref_ = device_;
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
