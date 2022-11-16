//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/TRootUniquePtr.root.cc
//---------------------------------------------------------------------------//
#include "TRootUniquePtr.hh"

#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
void TRootDeleter<T>::operator()(T* ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//
template struct TRootDeleter<TFile>;
template struct TRootDeleter<TTree>;
template struct TRootDeleter<TBranch>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
