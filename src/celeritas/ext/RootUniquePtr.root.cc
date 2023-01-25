//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootUniquePtr.root.cc
//---------------------------------------------------------------------------//
#include "RootUniquePtr.hh"

#include <TFile.h>
#include <TObject.h>
#include <TTree.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Call `TObject->Write()` before deletion. Used by TFile and TTree writer
 * classes.
 */
template<class T>
void WriteRootDeleter<T>::operator()(T* ptr)
{
    ptr->Write();
    delete ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Deleter only. Used by TFile and TTree reader classes or TObjects that should
 * not invoke `Write()`.
 */
template<class T>
void ReadRootDeleter<T>::operator()(T* ptr)
{
    delete ptr;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template struct WriteRootDeleter<TFile>;
template struct WriteRootDeleter<TTree>;

template struct ReadRootDeleter<TFile>;
template struct ReadRootDeleter<TTree>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
