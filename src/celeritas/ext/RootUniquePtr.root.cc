//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootUniquePtr.root.cc
//---------------------------------------------------------------------------//
#include "RootUniquePtr.hh"

#include <TFile.h>
#include <TTree.h>

#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call `TObject->Write()` before deletion. Used by TFile and TTree writer
 * classes. Can only be used if and only if a single ROOT file is open at any
 * given time.
 */
template<class T>
void RootWritableDeleter<T>::operator()(T* ptr)
{
    CELER_EXPECT(ptr);
    CELER_LOG(debug) << "Writing " << ptr->ClassName() << " '"
                     << ptr->GetName() << "'";
    ptr->Write();
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Call `TTree->AutoSave()` before deletion. Used by TTree when multiple tfiles
 * are open at the same time.
 */
template<class T>
void RootAutoSaveDeleter<T>::operator()(T* ptr)
{
    CELER_EXPECT(ptr);
    CELER_LOG(debug) << "Writing " << ptr->ClassName() << " '"
                     << ptr->GetName() << "'";
    ptr->AutoSave();
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Custom deleter to avoid propagating any dependency-specific implementation
 * downstream.
 */
template<class T>
void ExternDeleter<T>::operator()(T* ptr)
{
    CELER_EXPECT(ptr);
    CELER_LOG(debug) << "Closing " << ptr->ClassName() << " '"
                     << ptr->GetName() << "'";
    delete ptr;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//
template struct RootWritableDeleter<TFile>;
template struct RootWritableDeleter<TTree>;
template struct RootAutoSaveDeleter<TTree>;

template struct ExternDeleter<TFile>;
template struct ExternDeleter<TTree>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
