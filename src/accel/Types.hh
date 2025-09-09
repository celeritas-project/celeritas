//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/Types.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Setup mode for Celeritas usage
enum class OffloadMode
{
    uninitialized,  //!< Celeritas has not been initialized
    disabled,  //!< Offload is disabled
    kill_offload,  //!< Tracks are killed instead of offloaded
    enabled,  //!< Celeritas setup is complete
    size_
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
