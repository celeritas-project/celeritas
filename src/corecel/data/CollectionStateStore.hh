//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionStateStore.hh
//! \brief Backward compatibility header (deprecated)
//---------------------------------------------------------------------------//
#pragma once

#include "StateDataStore.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Deprecated alias for ParamsDataStore.
 *
 * \deprecated Use ParamsDataStore instead. This alias will be removed in v1.0.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
using CollectionStateStore
    [[deprecated("Use StateDataStore instead; will be removed in v1.0")]]
    = StateDataStore<S, M>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
