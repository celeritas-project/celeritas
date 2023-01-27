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
 * Custom deleter to avoid propagating any dependency-specific implementation
 * downstream the code.
 */
template<class T>
struct ExternDeleter
{
    void operator()(T* ptr);
};

//---------------------------------------------------------------------------//
// Type aliases
template<class T>
using UPRootWritable = std::unique_ptr<T, RootWritableDeleter<T>>;
template<class T>
using UPExtern = std::unique_ptr<T, ExternDeleter<T>>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
template<class T>
void RootWritableDeleter<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

template<class T>
void ExternDeleter<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
