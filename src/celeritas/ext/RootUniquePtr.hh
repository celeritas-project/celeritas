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

// Forward-declare ROOT; Expand as needed
class TObject;
class TFile;
class TTree;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helpers to prevent ROOT from propagating to downstream code.
 */
template<class T>
struct WriteAndDeleteRoot
{
    void operator()(T*);
};

template<class T>
struct DeleteRoot
{
    void operator()(T*) const;
};

template<class T>
using RootUPWrite = std::unique_ptr<T, WriteAndDeleteRoot<T>>;
template<class T>
using RootUPRead = std::unique_ptr<T, DeleteRoot<T>>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
template<class T>
void WriteAndDeleteRoot<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

template<class T>
void DeleteRoot<T>::operator()(T*) const
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
