//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootUniquePtr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

// Forward-declare ROOT
class TObject;

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Helpers to prevent ROOT from propagating to downstream code.
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * Call `TObject->Write()` before deletion. Used by TFile and TTree writer
 * classes.
 */
template<class T>
struct WriteRootDeleter
{
    void operator()(T* ptr);
};

//---------------------------------------------------------------------------//
/*!
 * Deleter only. Used by TFile and TTree reader classes or TObjects that should
 * not invoke `Write()`.
 */
template<class T>
struct ReadRootDeleter
{
    void operator()(T* ptr);
};

//---------------------------------------------------------------------------//
// Type aliases
template<class T>
using UPRootWriter = std::unique_ptr<T, WriteRootDeleter<T>>;

template<class T>
using UPRootReader = std::unique_ptr<T, ReadRootDeleter<T>>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
template<class T>
void WriteRootDeleter<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

template<class T>
void ReadRootDeleter<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
