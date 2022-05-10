//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/TFileUniquePtr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

// Forward-declare ROOT
class TFile;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Helper to prevent ROOT from propagating to downstream code.
struct TFileDeleter
{
    void operator()(TFile*) const;
};

using TFileUniquePtr = std::unique_ptr<TFile, TFileDeleter>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline void TFileDeleter::operator()(TFile*) const
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
