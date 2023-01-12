//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/RootSafeUniquePtrs.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

// Forward-declare ROOT
class TFile;
class TTree;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Helper to prevent ROOT from propagating to downstream code.
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//! TFile
struct TFileDeleter
{
    void operator()(TFile*) const;
};

using TFileSafeUniquePtr = std::unique_ptr<TFile, TFileDeleter>;

//---------------------------------------------------------------------------//
//! TTree
struct TTreeDeleter
{
    void operator()(TTree*) const;
};

using TTreeSafeUniquePtr = std::unique_ptr<TTree, TTreeDeleter>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline void TFileDeleter::operator()(TFile*) const
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void TTreeDeleter::operator()(TTree*) const
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
