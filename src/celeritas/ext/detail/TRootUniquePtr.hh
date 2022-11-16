//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/TRootUniquePtr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

// Forward-declare ROOT; expand as needed
class TFile;
class TTree;
class TBranch;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Helper to prevent ROOT from propagating to downstream code.
template<class T>
struct TRootDeleter
{
    void operator()(T*) const;
};

template<class T>
using TRootUniquePtr = std::unique_ptr<T, TRootDeleter<T>>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
template<class T>
inline void TRootDeleter<T>::operator()(T*) const
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
