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

namespace celeritas
{
template<class T>
void WriteAndDeleteRoot<T>::operator()(T* ptr)
{
    ptr->Write();
    delete ptr;
}

template<class T>
void DeleteRoot<T>::operator()(T* ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//
// Write and Delete
template struct WriteAndDeleteRoot<TFile>;
template struct WriteAndDeleteRoot<TTree>;

// Delete only
template struct DeleteRoot<TFile>;
template struct DeleteRoot<TTree>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
