//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootUniquePtr.hh
//! \brief Helpers to prevent ROOT from propagating to downstream code.
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call `TObject->Write()` before deletion. Used by TFile and TTree writer
 * classes.
 */
template<class T>
struct RootWritableDeleter
{
    void operator()(T* ptr);
};

//---------------------------------------------------------------------------//
/*!
 * Deleter only. Used by TFile and TTree reader classes or TObjects that should
 * not invoke `Write()`.
 */
template<class T>
struct RootReadDeleter
{
    void operator()(T* ptr);
};

//---------------------------------------------------------------------------//
// Type aliases
template<class T>
using UPRootWritable = std::unique_ptr<T, RootWritableDeleter<T>>;

template<class T>
using UPRootReadOnly = std::unique_ptr<T, RootReadDeleter<T>>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
template<class T>
void RootWritableDeleter<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

template<class T>
void RootReadDeleter<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
