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
 * classes.
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

template struct ExternDeleter<TFile>;
template struct ExternDeleter<TTree>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
