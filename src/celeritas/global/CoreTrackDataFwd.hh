//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackDataFwd.hh
//! \brief Forward declarations for some structs defined in CoreTrackData.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct CoreParamsData;

template<Ownership W, MemSpace M>
struct CoreStateData;

template<MemSpace M>
using CoreParamsPtr = CRefPtr<CoreParamsData, M>;
template<MemSpace M>
using CoreStatePtr = RefPtr<CoreStateData, M>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
