//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RunData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4Types.hh>

#include "celeritas/global/CoreParams.hh"
#include "accel/detail/LocalTransporter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared Celeritas params data and thread-local transporter.
 */
struct RunData
{
    using Transporter   = detail::LocalTransporter;
    using SPTransporter = std::shared_ptr<Transporter>;

    static G4ThreadLocal SPTransporter transport;
    std::shared_ptr<CoreParams>        params;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
