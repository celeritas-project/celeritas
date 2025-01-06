//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Call `TFile->Write()` before deletion.
 */
void RootFileWritableDeleter::operator()(TFile* ptr)
{
    CELER_EXPECT(ptr);
    CELER_LOG(debug) << "Writing " << ptr->ClassName() << " '"
                     << ptr->GetName() << "'";
    ptr->Write();
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Call `TTree->AutoSave()` before deletion. This ensures that `gDirectory` is
 * updated accordingly when multiple TFiles are open at the same time.
 */
void RootTreeAutoSaveDeleter::operator()(TTree* ptr)
{
    CELER_EXPECT(ptr);
    CELER_LOG(debug) << "Autosaving " << ptr->ClassName() << " '"
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
void RootExternDeleter<T>::operator()(T* ptr)
{
    CELER_EXPECT(ptr);
    CELER_LOG(debug) << "Closing " << ptr->ClassName() << " '"
                     << ptr->GetName() << "'";
    delete ptr;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//
template struct RootExternDeleter<TFile>;
template struct RootExternDeleter<TTree>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
