//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootUniquePtr.hh
//! \brief Helpers to prevent ROOT from propagating to downstream code.
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"  // IWYU pragma: keep

class TFile;
class TTree;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call `TFile->Write()` before deletion.
 */
struct RootFileWritableDeleter
{
    void operator()(TFile* ptr);
};

//---------------------------------------------------------------------------//
/*!
 * Call `TTree->AutoSave()` before deletion in order to update `gDirectory`
 * accordingly before writing the TTree.
 */
struct RootTreeAutoSaveDeleter
{
    void operator()(TTree* ptr);
};

//---------------------------------------------------------------------------//
/*!
 * Custom deleter to avoid propagating any dependency-specific implementation
 * downstream the code.
 */
template<class T>
struct RootExternDeleter
{
    void operator()(T* ptr);
};

//---------------------------------------------------------------------------//
// Type aliases
using UPRootFileWritable = std::unique_ptr<TFile, RootFileWritableDeleter>;
using UPRootTreeWritable = std::unique_ptr<TTree, RootTreeAutoSaveDeleter>;

template<class T>
using UPExtern = std::unique_ptr<T, RootExternDeleter<T>>;

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline void RootFileWritableDeleter::operator()(TFile*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootTreeAutoSaveDeleter::operator()(TTree*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

template<class T>
void RootExternDeleter<T>::operator()(T*)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
