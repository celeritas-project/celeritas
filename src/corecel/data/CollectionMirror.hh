//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionMirror.hh
//! \brief Backward compatibility header (deprecated)
//---------------------------------------------------------------------------//
#pragma once

#include "ParamsDataStore.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Deprecated alias for ParamsDataStore.
 *
 * \deprecated Use ParamsDataStore instead. This alias will be removed in v1.0.
 */
template<template<Ownership, MemSpace> class P>
using CollectionMirror
    [[deprecated("Use ParamsDataStore instead; will be removed in v1.0")]]
    = ParamsDataStore<P>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
